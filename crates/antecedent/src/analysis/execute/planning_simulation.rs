//! Whole-procedure simulation of a planned design: draw the trial the
//! assumptions declare, analyse it with the estimator the design is analysed
//! with, apply the planned decision, and replicate.
//!
//! Each replicate opens its own stream from a semantic key (the planning
//! operation and the replicate index), so a replicate's draws do not depend on
//! which worker runs it or how replicates are batched, and the replicate
//! records are reduced in index order: the result is the same at every thread
//! count. The estimators are the crate's own (`complete_unit_itt`,
//! `fit_ancova`, `blocked_unit_itt`, `complete_cluster_itt`, `factorial_2x2`,
//! `switchback_itt`); a replicate whose estimator returns no estimate counts as
//! no rejection, as an analysis that cannot be run rejects nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalRng, ChainId, ExecutionContext, FoldId, ReductionPolicy, ReplicateId, RngKey,
    SemanticOperation,
};
use antecedent_design::planning::{
    ArmComparison, FactorialContrast, FamilyPower, FamilySimulation, InterferenceExposure,
    Multiplicity, OutcomeModel, PlannedDesign, PlanningCriterion, PlanningRequest, PlanningResult,
    SequentialSimulation, SimulatedCharacteristics, SimulationSpec, SuccessCriterion, allocate,
};
use antecedent_estimate::ancova::fit_ancova;
use antecedent_estimate::randomized_neyman::{
    blocked_unit_itt, complete_cluster_itt, complete_unit_itt, factorial_2x2,
};
use antecedent_estimate::switchback::switchback_itt;
use antecedent_stats::power::critical_value;

use crate::error::CausalError;

/// Outcomes tagged with their treatment indicator.
type TaggedOutcomes = Vec<(f64, bool)>;

/// Semantic operation tag of every planning simulation stream.
const PLANNING_OPERATION: u64 = u64::from_le_bytes(*b"planning");

/// One replicate's outcome.
#[derive(Clone, Copy, Debug, Default)]
struct Record {
    /// The criterion's per-replicate value (an indicator, a rejected share, or
    /// a posterior standard deviation).
    value: f64,
    /// Estimated contrast; `NaN` when the estimator returned none.
    estimate: f64,
    /// At least one comparison rejected.
    any: bool,
    /// Every comparison rejected.
    all: bool,
    /// Share of comparisons rejected.
    share: f64,
    /// Look at which a sequential trial stopped (`u32::MAX`: ran to the end).
    stop_look: u32,
    /// Enrolled units of a sequential trial.
    units: u64,
}

/// One analysed contrast: estimate and its estimated variance.
#[derive(Clone, Copy, Debug)]
struct Analysed {
    estimate: f64,
    variance: f64,
    df: Option<f64>,
}

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn bernoulli(rng: &mut CausalRng, p: f64) -> bool {
    rng.next_f64() < p
}

/// Marsaglia–Tsang gamma draw of shape `shape` and unit scale.
fn gamma(rng: &mut CausalRng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = 1.0 - rng.next_f64();
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = 1.0 - rng.next_f64();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn beta(rng: &mut CausalRng, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a);
    let y = gamma(rng, b);
    x / (x + y)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a finite, nonnegative draw rounded to a count"
)]
fn count(x: f64) -> usize {
    x.round().max(0.0) as usize
}

/// Fisher–Yates shuffle.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a uniform in [0, 1) scaled to an index bound"
)]
fn shuffle<T>(rng: &mut CausalRng, values: &mut [T]) {
    for i in (1..values.len()).rev() {
        let j = ((rng.next_f64() * (i + 1) as f64) as usize).min(i);
        values.swap(i, j);
    }
}

/// The simulation of one validated, analytically planned request.
struct Simulation<'a> {
    request: &'a PlanningRequest,
    design: PlannedDesign,
    result: &'a PlanningResult,
}

impl Simulation<'_> {
    fn kept(&self) -> f64 {
        1.0 - self.request.assumptions.attrition
    }

    fn two_sided(&self) -> bool {
        self.request.assumptions.sidedness.is_two_sided()
    }

    fn alpha(&self) -> f64 {
        self.request.assumptions.alpha
    }

    fn rejects(&self, statistic: f64, critical: f64) -> bool {
        if self.two_sided() { statistic.abs() >= critical } else { statistic >= critical }
    }

    /// One outcome of an individual in an arm with effect `shift`, the
    /// covariates drawn into `covariates`.
    fn outcome(&self, rng: &mut CausalRng, shift: f64, covariates: &mut [f64]) -> f64 {
        let a = &self.request.assumptions;
        match a.outcome {
            OutcomeModel::Continuous { sd, control_mean } => {
                let level = control_mean.unwrap_or(0.0) + shift;
                if covariates.is_empty() {
                    return level + sd * normal(rng);
                }
                let per = (a.covariate_r_squared / covariates.len() as f64).sqrt();
                let mut explained = 0.0;
                for slot in covariates.iter_mut() {
                    *slot = normal(rng);
                    explained += per * *slot;
                }
                level + sd * (explained + (1.0 - a.covariate_r_squared).sqrt() * normal(rng))
            }
            OutcomeModel::Binary { control_rate } => {
                f64::from(u8::from(bernoulli(rng, (control_rate + shift).clamp(0.0, 1.0))))
            }
        }
    }

    /// Analysis of a two-arm sample: difference in means, or ANCOVA when the
    /// plan adjusts for covariates.
    fn two_arm(
        &self,
        outcomes: &[f64],
        treated: &[bool],
        covariates: &[Vec<f64>],
    ) -> Option<Analysed> {
        let k = self.request.assumptions.covariates as usize;
        let n = outcomes.len();
        let binary = self.request.assumptions.outcome.is_binary();
        if k == 0 {
            let (t, c): (TaggedOutcomes, TaggedOutcomes) =
                outcomes.iter().copied().zip(treated.iter().copied()).partition(|(_, w)| *w);
            let t: Vec<f64> = t.into_iter().map(|(y, _)| y).collect();
            let c: Vec<f64> = c.into_iter().map(|(y, _)| y).collect();
            let fit = complete_unit_itt(&t, &c)?;
            return Some(Analysed {
                estimate: fit.effect,
                variance: fit.variance_upper_bound,
                df: (!binary).then_some(n as f64 - 2.0),
            });
        }
        let columns: Vec<&[f64]> = covariates.iter().map(Vec::as_slice).collect();
        let fit = fit_ancova(outcomes, treated, &columns).ok()?;
        Some(Analysed {
            estimate: fit.effect,
            variance: fit.hc0_variance,
            df: Some(n as f64 - 2.0 - k as f64),
        })
    }

    /// Draw `units` individuals with the given assignment, drop attrition, and
    /// analyse the two-arm contrast.
    fn two_arm_trial(
        &self,
        rng: &mut CausalRng,
        assignment: &[bool],
        effect: f64,
    ) -> Option<Analysed> {
        let k = self.request.assumptions.covariates as usize;
        let mut outcomes = Vec::with_capacity(assignment.len());
        let mut treated = Vec::with_capacity(assignment.len());
        let mut covariates = vec![Vec::with_capacity(assignment.len()); k];
        let mut row = vec![0.0; k];
        let kept = self.kept();
        for &w in assignment {
            let y = self.outcome(rng, if w { effect } else { 0.0 }, &mut row);
            if kept < 1.0 && !bernoulli(rng, kept) {
                continue;
            }
            outcomes.push(y);
            treated.push(w);
            for (column, value) in covariates.iter_mut().zip(&row) {
                column.push(*value);
            }
        }
        self.two_arm(&outcomes, &treated, &covariates)
    }

    fn arms(&self) -> (u64, u64) {
        let arms = &self.result.arm_units;
        (arms[0], arms[1])
    }

    fn fixed_two_arm(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let (c, t) = self.arms();
        let mut assignment = vec![false; usize::try_from(c).unwrap_or(0)];
        assignment.extend(std::iter::repeat_n(true, usize::try_from(t).unwrap_or(0)));
        self.two_arm_trial(rng, &assignment, effect)
    }

    fn bernoulli_trial(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let r = self.request.assumptions.allocation_ratio;
        let p = r / (1.0 + r);
        let assignment: Vec<bool> = (0..self.result.units).map(|_| bernoulli(rng, p)).collect();
        self.two_arm_trial(rng, &assignment, effect)
    }

    fn arm_sample(&self, rng: &mut CausalRng, size: u64, shift: f64, sd: Option<f64>) -> Vec<f64> {
        let kept = self.kept();
        let mut out = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
        for _ in 0..size {
            let y = match sd {
                Some(sd) => shift + sd * normal(rng),
                None => self.outcome(rng, shift, &mut []),
            };
            if kept >= 1.0 || bernoulli(rng, kept) {
                out.push(y);
            }
        }
        out
    }

    fn stratified_trial(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let PlannedDesign::Stratified { strata } = &self.design else { return None };
        let shares: Vec<f64> = strata.iter().map(|s| s.share).collect();
        let r = self.request.assumptions.allocation_ratio;
        let mut samples = Vec::with_capacity(strata.len());
        for (stratum, size) in strata.iter().zip(allocate(self.result.units, &shares)) {
            let arms = allocate(size, &[1.0, r]);
            let (control, treated) = match self.request.assumptions.outcome {
                OutcomeModel::Continuous { .. } => (
                    self.arm_sample(rng, arms[0], 0.0, Some(stratum.within)),
                    self.arm_sample(rng, arms[1], effect, Some(stratum.within)),
                ),
                OutcomeModel::Binary { .. } => {
                    let draw = |rng: &mut CausalRng, n: u64, p: f64| -> Vec<f64> {
                        let kept = self.kept();
                        (0..n)
                            .filter_map(|_| {
                                let y = f64::from(u8::from(bernoulli(rng, p.clamp(0.0, 1.0))));
                                (kept >= 1.0 || bernoulli(rng, kept)).then_some(y)
                            })
                            .collect()
                    };
                    (
                        draw(rng, arms[0], stratum.within),
                        draw(rng, arms[1], stratum.within + effect),
                    )
                }
            };
            samples.push((treated, control));
        }
        let blocks: Vec<(&[f64], &[f64])> =
            samples.iter().map(|(t, c)| (t.as_slice(), c.as_slice())).collect();
        let fit = blocked_unit_itt(&blocks)?;
        let rows: usize = samples.iter().map(|(t, c)| t.len() + c.len()).sum();
        let df = (!self.request.assumptions.outcome.is_binary())
            .then_some(rows as f64 - 2.0 * strata.len() as f64);
        Some(Analysed { estimate: fit.effect, variance: fit.variance_upper_bound, df })
    }

    fn cluster_trial(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let (m, icc, cv) = match &self.design {
            PlannedDesign::Cluster { mean_cluster_size, icc, cluster_size_cv }
            | PlannedDesign::Interference {
                exposure:
                    Some(InterferenceExposure::ContainedInClusters {
                        mean_cluster_size,
                        icc,
                        cluster_size_cv,
                    }),
            } => (*mean_cluster_size, *icc, *cluster_size_cv),
            _ => return None,
        };
        let (kc, kt) = self.arms();
        let kept = self.kept();
        let mut rows = 0_usize;
        let mut totals = [Vec::new(), Vec::new()];
        for (arm, clusters) in [(0_usize, kc), (1, kt)] {
            let shift = if arm == 1 { effect } else { 0.0 };
            for _ in 0..clusters {
                let size = if cv > 0.0 {
                    let shape = 1.0 / (cv * cv);
                    count(gamma(rng, shape) * m / shape).max(1)
                } else {
                    count(m).max(1)
                };
                let mut total = 0.0;
                match self.request.assumptions.outcome {
                    OutcomeModel::Continuous { sd, control_mean } => {
                        let level = control_mean.unwrap_or(0.0) + shift;
                        let random = sd * icc.sqrt() * normal(rng);
                        let within = sd * (1.0 - icc).sqrt();
                        for _ in 0..size {
                            let y = level + random + within * normal(rng);
                            if kept >= 1.0 || bernoulli(rng, kept) {
                                total += y;
                                rows += 1;
                            }
                        }
                    }
                    OutcomeModel::Binary { control_rate } => {
                        let p = (control_rate + shift).clamp(1e-12, 1.0 - 1e-12);
                        let cluster_p = if icc > 0.0 {
                            let scale = (1.0 - icc) / icc;
                            beta(rng, p * scale, (1.0 - p) * scale)
                        } else {
                            p
                        };
                        for _ in 0..size {
                            let y = f64::from(u8::from(bernoulli(rng, cluster_p)));
                            if kept >= 1.0 || bernoulli(rng, kept) {
                                total += y;
                                rows += 1;
                            }
                        }
                    }
                }
                totals[arm].push(total);
            }
        }
        let fit = complete_cluster_itt(&totals[1], &totals[0], rows)?;
        let df = (!self.request.assumptions.outcome.is_binary()).then_some((kc + kt) as f64 - 2.0);
        Some(Analysed { estimate: fit.effect, variance: fit.variance_upper_bound, df })
    }

    fn factorial_trial(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let PlannedDesign::Factorial { contrast } = &self.design else { return None };
        let cells = &self.result.arm_units;
        let treated: &[usize] = match contrast {
            FactorialContrast::MainEffect => &[1, 3],
            FactorialContrast::Interaction => &[3],
        };
        let samples: Vec<Vec<f64>> = (0..4)
            .map(|index| {
                let shift = if treated.contains(&index) { effect } else { 0.0 };
                self.arm_sample(rng, cells[index], shift, None)
            })
            .collect();
        let fit = factorial_2x2([
            samples[0].as_slice(),
            samples[1].as_slice(),
            samples[2].as_slice(),
            samples[3].as_slice(),
        ])?;
        let rows: usize = samples.iter().map(Vec::len).sum();
        let df = (!self.request.assumptions.outcome.is_binary()).then_some(rows as f64 - 4.0);
        Some(match contrast {
            FactorialContrast::MainEffect => Analysed {
                estimate: fit.primary,
                variance: fit.main_effect_variance_upper_bound,
                df,
            },
            FactorialContrast::Interaction => Analysed {
                estimate: fit.interaction,
                variance: fit.interaction_variance_upper_bound,
                df,
            },
        })
    }

    fn switchback_trial(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        let PlannedDesign::Switchback {
            sequences,
            periods_per_block,
            autocorrelation,
            carryover,
            washout_periods,
        } = &self.design
        else {
            return None;
        };
        let OutcomeModel::Continuous { sd, control_mean } = self.request.assumptions.outcome else {
            return None;
        };
        let mu = control_mean.unwrap_or(0.0);
        let r = self.request.assumptions.allocation_ratio;
        let p = r / (1.0 + r);
        let direct = effect / (1.0 + carryover);
        let innovation = sd * (1.0 - autocorrelation * autocorrelation).sqrt();
        let blocks = self.result.units;
        let mut outcomes = Vec::new();
        let mut assignment = Vec::new();
        let labels: Vec<String> = (0..*sequences).map(|sequence| format!("s{sequence}")).collect();
        let mut ids: Vec<&str> = Vec::new();
        for label in &labels {
            let mut noise = sd * normal(rng);
            let mut previous = bernoulli(rng, p);
            let mut first = true;
            for _ in 0..blocks {
                let w = bernoulli(rng, p);
                for period in 0..*periods_per_block {
                    if !first {
                        noise = autocorrelation * noise + innovation * normal(rng);
                    }
                    first = false;
                    let y = mu
                        + if w { direct } else { 0.0 }
                        + if previous { carryover * direct } else { 0.0 }
                        + noise;
                    previous = w;
                    if period >= *washout_periods {
                        outcomes.push(y);
                        assignment.push(w);
                        ids.push(label.as_str());
                    }
                }
            }
        }
        let probabilities = vec![p; outcomes.len()];
        let fit = switchback_itt(&outcomes, &assignment, &probabilities, &ids)?;
        Some(Analysed {
            estimate: fit.effect,
            variance: fit.variance,
            df: Some(*sequences as f64 - 1.0),
        })
    }

    /// Analysed contrast of a single-contrast design at `effect`.
    fn single(&self, rng: &mut CausalRng, effect: f64) -> Option<Analysed> {
        match &self.design {
            PlannedDesign::Bernoulli => self.bernoulli_trial(rng, effect),
            PlannedDesign::Complete => self.fixed_two_arm(rng, effect),
            PlannedDesign::Stratified { .. } => self.stratified_trial(rng, effect),
            PlannedDesign::Cluster { .. } | PlannedDesign::Interference { .. } => {
                self.cluster_trial(rng, effect)
            }
            PlannedDesign::Factorial { .. } => self.factorial_trial(rng, effect),
            PlannedDesign::Switchback { .. } => self.switchback_trial(rng, effect),
            PlannedDesign::MultiArm { .. }
            | PlannedDesign::GroupSequential { .. }
            | PlannedDesign::OutcomeAdaptive { .. } => None,
        }
    }

    fn single_rejects(&self, analysed: Option<Analysed>) -> bool {
        analysed.is_some_and(|a| {
            a.variance > 0.0
                && self.rejects(
                    a.estimate / a.variance.sqrt(),
                    critical_value(self.alpha(), a.df, self.two_sided()),
                )
        })
    }

    fn multi_arm(&self, rng: &mut CausalRng, effect: f64) -> Record {
        let PlannedDesign::MultiArm { treatment_arms, comparison } = &self.design else {
            return Record::default();
        };
        let arms = &self.result.arm_units;
        let samples: Vec<Vec<f64>> = arms
            .iter()
            .enumerate()
            .map(|(index, size)| {
                self.arm_sample(rng, *size, if index == 0 { 0.0 } else { effect }, None)
            })
            .collect();
        let k = *treatment_arms;
        let rows: usize = samples.iter().map(Vec::len).sum();
        let binary = self.request.assumptions.outcome.is_binary();
        let df = (!binary).then_some(rows as f64 - (k + 1) as f64);
        let first = complete_unit_itt(&samples[1], &samples[0]);
        let estimate = first.map_or(f64::NAN, |fit| fit.effect);
        match comparison {
            ArmComparison::Omnibus => {
                let rejected = omnibus_statistic(&samples).is_some_and(|f| {
                    f >= self.result.analytic.critical_value.unwrap_or(f64::INFINITY)
                });
                Record {
                    value: f64::from(u8::from(rejected)),
                    estimate,
                    any: rejected,
                    all: rejected,
                    share: f64::from(u8::from(rejected)),
                    ..Record::default()
                }
            }
            ArmComparison::ManyToOne { multiplicity, power } => {
                let critical: Vec<f64> = self.result.analytic.family.as_ref().map_or_else(
                    || vec![critical_value(self.alpha() / k as f64, df, self.two_sided()); k],
                    |family| family.critical.clone(),
                );
                let statistics: Vec<f64> = samples[1..]
                    .iter()
                    .map(|arm| {
                        complete_unit_itt(arm, &samples[0])
                            .filter(|fit| fit.variance_upper_bound > 0.0)
                            .map_or(f64::NAN, |fit| fit.effect / fit.variance_upper_bound.sqrt())
                    })
                    .collect();
                let rejected = match multiplicity {
                    Multiplicity::Bonferroni | Multiplicity::Dunnett => statistics
                        .iter()
                        .filter(|t| !t.is_nan() && self.rejects(**t, critical[0]))
                        .count(),
                    Multiplicity::Holm => {
                        let mut ordered: Vec<f64> = statistics
                            .iter()
                            .map(|t| {
                                if t.is_nan() {
                                    f64::NEG_INFINITY
                                } else if self.two_sided() {
                                    t.abs()
                                } else {
                                    *t
                                }
                            })
                            .collect();
                        ordered.sort_by(|a, b| b.total_cmp(a));
                        ordered.iter().zip(&critical).take_while(|(t, c)| **t >= **c).count()
                    }
                };
                let share = rejected as f64 / k as f64;
                let (any, all) = (rejected > 0, rejected == k);
                let value = match power {
                    FamilyPower::PerComparison => share,
                    FamilyPower::Any => f64::from(u8::from(any)),
                    FamilyPower::All => f64::from(u8::from(all)),
                };
                Record { value, estimate, any, all, share, ..Record::default() }
            }
        }
    }

    fn sequential(&self, rng: &mut CausalRng, effect: f64) -> Record {
        let Some(sequential) = self.result.analytic.sequential.as_ref() else {
            return Record::default();
        };
        let (c, t) = self.arms();
        let mut assignment = vec![false; usize::try_from(c).unwrap_or(0)];
        assignment.extend(std::iter::repeat_n(true, usize::try_from(t).unwrap_or(0)));
        shuffle(rng, &mut assignment);
        let kept = self.kept();
        let mut outcomes = Vec::with_capacity(assignment.len());
        for &w in &assignment {
            let y = self.outcome(rng, if w { effect } else { 0.0 }, &mut []);
            let observed = kept >= 1.0 || bernoulli(rng, kept);
            outcomes.push((y, w, observed));
        }
        let mut estimate = f64::NAN;
        for (look, (&units, &critical)) in
            sequential.look_units.iter().zip(&sequential.critical).enumerate()
        {
            let upto = usize::try_from(units).unwrap_or(usize::MAX).min(outcomes.len());
            let (mut treated, mut control) = (Vec::new(), Vec::new());
            for &(y, w, observed) in &outcomes[..upto] {
                if observed {
                    if w { treated.push(y) } else { control.push(y) }
                }
            }
            let Some(fit) = complete_unit_itt(&treated, &control) else { continue };
            estimate = fit.effect;
            if fit.variance_upper_bound > 0.0
                && self.rejects(fit.effect / fit.variance_upper_bound.sqrt(), critical)
            {
                return Record {
                    value: 1.0,
                    estimate,
                    any: true,
                    all: true,
                    share: 1.0,
                    stop_look: u32::try_from(look).unwrap_or(u32::MAX),
                    units,
                };
            }
        }
        Record { estimate, stop_look: u32::MAX, units: self.result.units, ..Record::default() }
    }

    /// Draw the effect of a replicate: the assumed (or solved) effect, or a
    /// draw from the design prior.
    fn effect(&self, rng: &mut CausalRng) -> f64 {
        let draw = |rng: &mut CausalRng, mean: f64, sd: f64| {
            if sd > 0.0 { mean + sd * normal(rng) } else { mean }
        };
        let theta = match self.request.criterion {
            PlanningCriterion::Power { .. } => self.result.effect.unwrap_or(0.0),
            PlanningCriterion::Assurance { design_prior, .. }
            | PlanningCriterion::CorrectDecision { design_prior, .. } => {
                draw(rng, design_prior.mean, design_prior.sd)
            }
            PlanningCriterion::PosteriorPrecision { analysis_prior, .. } => analysis_prior.mean,
        };
        match self.request.assumptions.outcome {
            OutcomeModel::Binary { control_rate } => {
                theta.clamp(1e-9 - control_rate, 1.0 - 1e-9 - control_rate)
            }
            OutcomeModel::Continuous { .. } => theta,
        }
    }

    fn replicate(&self, index: u64, seed: u64) -> Record {
        let key = RngKey::new(
            SemanticOperation::new(PLANNING_OPERATION),
            FoldId::new(0),
            ChainId::new(0),
            ReplicateId::new(index),
            ReductionPolicy::Strict,
        );
        let mut rng = key.open(seed);
        let theta = self.effect(&mut rng);
        match &self.design {
            PlannedDesign::MultiArm { .. } => return self.multi_arm(&mut rng, theta),
            PlannedDesign::GroupSequential { .. } => return self.sequential(&mut rng, theta),
            _ => {}
        }
        let analysed = self.single(&mut rng, theta);
        let estimate = analysed.map_or(f64::NAN, |a| a.estimate);
        let value = match self.request.criterion {
            PlanningCriterion::Power { .. }
            | PlanningCriterion::Assurance { success: SuccessCriterion::Significance, .. } => {
                f64::from(u8::from(self.single_rejects(analysed)))
            }
            PlanningCriterion::Assurance {
                success: SuccessCriterion::Posterior { analysis_prior, threshold, probability },
                ..
            } => f64::from(u8::from(declares(analysed, analysis_prior, threshold, probability))),
            PlanningCriterion::PosteriorPrecision { analysis_prior, .. } => {
                let prior_precision = 1.0 / (analysis_prior.sd * analysis_prior.sd);
                let data = analysed.filter(|a| a.variance > 0.0).map_or(0.0, |a| 1.0 / a.variance);
                1.0 / (prior_precision + data).sqrt()
            }
            PlanningCriterion::CorrectDecision {
                analysis_prior, threshold, probability, ..
            } => {
                let declared = declares(analysed, analysis_prior, threshold, probability);
                f64::from(u8::from(declared == (theta > threshold)))
            }
        };
        let rejected = value > 0.5;
        Record {
            value,
            estimate,
            any: rejected,
            all: rejected,
            share: value,
            stop_look: u32::MAX,
            units: self.result.units,
        }
    }
}

/// Whether the conjugate posterior of the analysed contrast puts at least
/// `probability` above `threshold`. No estimate declares nothing.
fn declares(
    analysed: Option<Analysed>,
    prior: antecedent_design::planning::NormalPrior,
    threshold: f64,
    probability: f64,
) -> bool {
    let prior_precision = 1.0 / (prior.sd * prior.sd);
    let Some(a) = analysed.filter(|a| a.variance > 0.0) else {
        return false;
    };
    let precision = prior_precision + 1.0 / a.variance;
    let mean = (prior.mean * prior_precision + a.estimate / a.variance) / precision;
    antecedent_kernels::norm_sf((threshold - mean) * precision.sqrt()) >= probability
}

/// One-way analysis-of-variance F statistic of the arm samples.
fn omnibus_statistic(samples: &[Vec<f64>]) -> Option<f64> {
    let rows: usize = samples.iter().map(Vec::len).sum();
    let groups = samples.len();
    if groups < 2 || rows <= groups || samples.iter().any(Vec::is_empty) {
        return None;
    }
    let grand = samples.iter().flatten().sum::<f64>() / rows as f64;
    let (mut between, mut within) = (0.0, 0.0);
    for sample in samples {
        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        between += sample.len() as f64 * (mean - grand).powi(2);
        within += sample.iter().map(|y| (y - mean).powi(2)).sum::<f64>();
    }
    let statistic = (between / (groups - 1) as f64) / (within / (rows - groups) as f64);
    statistic.is_finite().then_some(statistic)
}

/// Bytes one worker's data scratch needs at the plan's size.
pub(crate) fn scratch_bytes(request: &PlanningRequest, result: &PlanningResult) -> u64 {
    let per_individual = 48 + 8 * u64::from(request.assumptions.covariates);
    let inflation = match &request.design {
        PlannedDesign::Cluster { cluster_size_cv, .. } => 1.0 + 4.0 * cluster_size_cv,
        _ => 1.0,
    };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite, nonnegative byte bound saturates"
    )]
    let individuals = (result.enrolled_individuals * inflation).ceil().max(0.0) as u64;
    individuals.saturating_mul(per_individual).saturating_add(4096)
}

fn mean_and_se(values: impl Iterator<Item = f64> + Clone) -> (f64, f64, f64) {
    let n = values.clone().count();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean = values.clone().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::NAN, f64::NAN);
    }
    let variance = values.map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, variance.sqrt(), (variance / n as f64).sqrt())
}

/// Simulate the plan `result` of `request`: `spec.replicates` keyed replicates
/// under `ctx`, reduced in replicate order.
///
/// # Errors
///
/// Cancellation, or a worker failure.
pub(crate) fn simulate(
    request: &PlanningRequest,
    result: &PlanningResult,
    spec: SimulationSpec,
    ctx: &ExecutionContext,
) -> Result<SimulatedCharacteristics, CausalError> {
    let simulation = Simulation { request, design: request.effective_design(), result };
    let replicates = spec.replicates as usize;
    let records = ctx.map_indexed(replicates, |index, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "planning_simulation" });
        }
        Ok(simulation.replicate(index as u64, spec.seed))
    })?;
    let (value, _, monte_carlo_se) = mean_and_se(records.iter().map(|r| r.value));
    let estimates = records.iter().map(|r| r.estimate).filter(|e| e.is_finite());
    let (mean_estimate, estimate_sd, _) = mean_and_se(estimates);
    let failed = records.iter().filter(|r| !r.estimate.is_finite()).count();
    let n = records.len() as f64;
    let family =
        matches!(request.design, PlannedDesign::MultiArm { .. }).then(|| FamilySimulation {
            any: records.iter().filter(|r| r.any).count() as f64 / n,
            all: records.iter().filter(|r| r.all).count() as f64 / n,
            per_comparison: records.iter().map(|r| r.share).sum::<f64>() / n,
        });
    let sequential = result.analytic.sequential.as_ref().map(|plan| {
        let mut stop_frequency = vec![0.0; plan.look_units.len()];
        for record in &records {
            if let Some(slot) = stop_frequency.get_mut(record.stop_look as usize) {
                *slot += 1.0 / n;
            }
        }
        let (expected_units, _, expected_units_se) =
            mean_and_se(records.iter().map(|r| r.units as f64));
        SequentialSimulation { stop_frequency, expected_units, expected_units_se }
    });
    let agreement_z =
        (monte_carlo_se > 0.0).then(|| (value - result.analytic.value) / monte_carlo_se);
    Ok(SimulatedCharacteristics {
        value,
        monte_carlo_se,
        replicates: spec.replicates,
        seed: spec.seed,
        failed_replicates: u32::try_from(failed).unwrap_or(u32::MAX),
        mean_estimate,
        estimate_sd,
        family,
        sequential,
        agreement_z,
    })
}
