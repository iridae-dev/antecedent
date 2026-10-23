//! Analytic operating characteristics of each planned design, and the solves
//! in both directions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::collections::BTreeMap;

use antecedent_kernels::{norm_cdf, norm_inv, norm_sf};
use antecedent_stats::adaptive_inference::group_sequential::{
    Sidedness, boundaries, crossing_probabilities,
};
use antecedent_stats::ln_gamma;
use antecedent_stats::power::{
    EquicorrelatedComparisons, critical_value, f_test_power, t_test_power,
};
use antecedent_stats::special::{gauss_hermite_standard_normal, gauss_legendre};

use super::request::{
    ArmComparison, FactorialContrast, FamilyPower, Multiplicity, OutcomeModel, PlannedDesign,
    PlanningCriterion, PlanningGoal, PlanningRefusal, PlanningRequest, SuccessCriterion,
};
use super::result::{
    AnalyticCharacteristics, FamilyCharacteristics, PlanningResult, SequentialCharacteristics,
};
use super::{PLANNING_CALIBRATION, RELAX_TARGET};

/// Largest size scanned for the design's smallest feasible size.
const MINIMUM_SCAN: u64 = 100_000;
/// Above this many units a Bernoulli design is evaluated at its expected arm sizes.
const BERNOULLI_MIXTURE_LIMIT: u64 = 50_000;
/// Gauss–Hermite nodes of a prior average.
const PRIOR_NODES: usize = 40;
/// Effect bisections of a minimum detectable effect.
const EFFECT_BISECTIONS: usize = 200;
/// Rates are kept this far inside `(0, 1)` when a prior moves them.
const RATE_GUARD: f64 = 1e-9;

/// One realized layout of the design and the variance of its contrast.
#[derive(Clone, Copy, Debug)]
struct Component {
    weight: f64,
    variance: f64,
    df: Option<f64>,
}

/// Law of a single planned contrast at one size and effect.
struct ContrastLaw {
    components: Vec<Component>,
    design_effect: Option<f64>,
}

impl ContrastLaw {
    fn single(variance: f64, df: Option<f64>, design_effect: Option<f64>) -> Self {
        Self { components: vec![Component { weight: 1.0, variance, df }], design_effect }
    }

    /// Mixture-weighted mean variance over components with a defined estimate.
    fn mean_variance(&self) -> f64 {
        let (mut total, mut mass) = (0.0, 0.0);
        for c in &self.components {
            if c.variance.is_finite() {
                total += c.weight * c.variance;
                mass += c.weight;
            }
        }
        if mass > 0.0 { total / mass } else { f64::INFINITY }
    }

    fn reference_df(&self) -> Option<f64> {
        self.components
            .iter()
            .filter(|c| c.variance.is_finite())
            .max_by(|a, b| a.weight.total_cmp(&b.weight))
            .and_then(|c| c.df)
    }
}

/// Power of the planned test and what produced it.
struct PowerDetail {
    power: f64,
    variance: f64,
    noncentrality: Option<f64>,
    df: Option<f64>,
    critical: Option<f64>,
    design_effect: Option<f64>,
    family: Option<FamilyCharacteristics>,
    sequential: Option<SequentialCharacteristics>,
    method: &'static str,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass finite nonnegative values"
)]
fn floor_u64(x: f64) -> u64 {
    x.floor().max(0.0) as u64
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass finite nonnegative values"
)]
fn round_u64(x: f64) -> u64 {
    x.round().max(0.0) as u64
}

/// Split `total` units over `weights` by the largest-remainder rule; ties go to
/// the earlier entry. Every planned layout (arms, cells, strata) is allocated
/// by this rule, and the simulation allocates the same way.
#[must_use]
pub fn allocate(total: u64, weights: &[f64]) -> Vec<u64> {
    let sum: f64 = weights.iter().sum();
    let raw: Vec<f64> = weights.iter().map(|w| total as f64 * w / sum).collect();
    let mut out: Vec<u64> = raw.iter().map(|r| floor_u64(*r)).collect();
    let assigned: u64 = out.iter().sum();
    let mut left = total.saturating_sub(assigned);
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = raw[a] - raw[a].floor();
        let fb = raw[b] - raw[b].floor();
        fb.total_cmp(&fa).then(a.cmp(&b))
    });
    for index in order {
        if left == 0 {
            break;
        }
        out[index] += 1;
        left -= 1;
    }
    out
}

fn rate(p: f64) -> f64 {
    p.clamp(RATE_GUARD, 1.0 - RATE_GUARD)
}

/// `1 + 2 Σ_{k<L} (1 − k/L) φ^k`: variance inflation of a sum of `L`
/// consecutive AR(1) periods.
fn block_inflation(periods: u64, autocorrelation: f64) -> f64 {
    let l = periods as f64;
    let mut total = 1.0;
    let mut power = 1.0;
    for k in 1..periods {
        power *= autocorrelation;
        total += 2.0 * (1.0 - k as f64 / l) * power;
    }
    total
}

/// Binomial probability mass `P(K = k)` for `K ~ Bin(n, p)`.
fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    let (nf, kf) = (n as f64, k as f64);
    (ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0)
        + kf * p.ln()
        + (nf - kf) * (-p).ln_1p())
    .exp()
}

/// The planner of one validated request.
struct Planner<'a> {
    request: &'a PlanningRequest,
    design: PlannedDesign,
    dunnett: RefCell<BTreeMap<(u64, u64), f64>>,
    boundaries: RefCell<BTreeMap<Vec<u64>, Vec<f64>>>,
}

impl<'a> Planner<'a> {
    fn new(request: &'a PlanningRequest) -> Self {
        Self {
            request,
            design: request.effective_design(),
            dunnett: RefCell::new(BTreeMap::new()),
            boundaries: RefCell::new(BTreeMap::new()),
        }
    }

    fn alpha(&self) -> f64 {
        self.request.assumptions.alpha
    }

    fn two_sided(&self) -> bool {
        self.request.assumptions.sidedness.is_two_sided()
    }

    fn ratio(&self) -> f64 {
        self.request.assumptions.allocation_ratio
    }

    fn kept(&self) -> f64 {
        1.0 - self.request.assumptions.attrition
    }

    fn binary(&self) -> bool {
        self.request.assumptions.outcome.is_binary()
    }

    /// `(control, treated)` outcome variances at `effect`.
    fn arm_variances(&self, effect: f64) -> (f64, f64) {
        let a = &self.request.assumptions;
        match a.outcome {
            OutcomeModel::Continuous { sd, .. } => {
                let v = sd * sd * (1.0 - a.covariate_r_squared);
                (v, v)
            }
            OutcomeModel::Binary { control_rate } => {
                let (p0, p1) = (rate(control_rate), rate(control_rate + effect));
                (p0 * (1.0 - p0), p1 * (1.0 - p1))
            }
        }
    }

    fn control_level(&self) -> f64 {
        match self.request.assumptions.outcome {
            OutcomeModel::Continuous { control_mean, .. } => control_mean.unwrap_or(0.0),
            OutcomeModel::Binary { control_rate } => control_rate,
        }
    }

    fn df(&self, value: f64) -> Option<f64> {
        (!self.binary()).then_some(value)
    }

    fn two_arm(&self, units: u64) -> (u64, u64) {
        let arms = allocate(units, &[1.0, self.ratio()]);
        (arms[0], arms[1])
    }

    fn treatment_probability(&self) -> f64 {
        self.ratio() / (1.0 + self.ratio())
    }

    /// Enrolled units per arm or cell, control first.
    fn arm_units(&self, units: u64) -> Vec<u64> {
        match &self.design {
            PlannedDesign::Bernoulli => {
                let treated = round_u64(units as f64 * self.treatment_probability());
                vec![units - treated.min(units), treated.min(units)]
            }
            PlannedDesign::Complete
            | PlannedDesign::Cluster { .. }
            | PlannedDesign::GroupSequential { .. } => {
                let (c, t) = self.two_arm(units);
                vec![c, t]
            }
            PlannedDesign::Stratified { strata } => {
                let sizes = allocate(units, &strata.iter().map(|s| s.share).collect::<Vec<_>>());
                let mut arms = vec![0, 0];
                for size in sizes {
                    let (c, t) = self.two_arm(size);
                    arms[0] += c;
                    arms[1] += t;
                }
                arms
            }
            PlannedDesign::MultiArm { treatment_arms, .. } => {
                let mut weights = vec![1.0];
                weights.extend(std::iter::repeat_n(self.ratio(), *treatment_arms));
                allocate(units, &weights)
            }
            PlannedDesign::Factorial { .. } => allocate(units, &[1.0; 4]),
            PlannedDesign::Switchback { .. }
            | PlannedDesign::Interference { .. }
            | PlannedDesign::OutcomeAdaptive { .. } => Vec::new(),
        }
    }

    fn enrolled_individuals(&self, units: u64) -> f64 {
        match &self.design {
            PlannedDesign::Cluster { mean_cluster_size, .. } => units as f64 * mean_cluster_size,
            PlannedDesign::Switchback { sequences, periods_per_block, .. } => {
                units.saturating_mul(*sequences).saturating_mul(*periods_per_block) as f64
            }
            _ => units as f64,
        }
    }

    fn analysed_individuals(&self, units: u64) -> f64 {
        match &self.design {
            PlannedDesign::Switchback { sequences, periods_per_block, washout_periods, .. } => {
                units.saturating_mul(*sequences).saturating_mul(periods_per_block - washout_periods)
                    as f64
            }
            _ => self.enrolled_individuals(units) * self.kept(),
        }
    }

    fn feasible(&self, units: u64) -> bool {
        let kept = self.kept();
        let enough = |n: u64| n as f64 * kept >= 2.0;
        match &self.design {
            PlannedDesign::Bernoulli => {
                let p = self.treatment_probability();
                units >= 4 && units as f64 * kept * p.min(1.0 - p) >= 2.0
            }
            PlannedDesign::Complete => {
                let (c, t) = self.two_arm(units);
                enough(c)
                    && enough(t)
                    && units as f64 * kept > 2.0 + f64::from(self.request.assumptions.covariates)
            }
            PlannedDesign::Stratified { strata } => {
                let sizes = allocate(units, &strata.iter().map(|s| s.share).collect::<Vec<_>>());
                sizes.into_iter().all(|size| {
                    let (c, t) = self.two_arm(size);
                    enough(c) && enough(t)
                })
            }
            PlannedDesign::Cluster { .. } => {
                let (c, t) = self.two_arm(units);
                c >= 2 && t >= 2
            }
            PlannedDesign::MultiArm { treatment_arms, .. } => {
                let arms = self.arm_units(units);
                arms.iter().all(|n| enough(*n))
                    && units as f64 * kept > (*treatment_arms + 1) as f64
            }
            PlannedDesign::Factorial { .. } => {
                allocate(units, &[1.0; 4]).into_iter().all(enough) && units as f64 * kept > 4.0
            }
            PlannedDesign::Switchback { .. } => units >= 1,
            PlannedDesign::GroupSequential { information_fractions, .. } => {
                let looks = Self::look_units(units, information_fractions);
                let increasing = looks.windows(2).all(|w| w[1] > w[0]);
                let first = looks.first().copied().unwrap_or(0);
                let (c, t) = self.two_arm(first);
                increasing && enough(c) && enough(t)
            }
            PlannedDesign::Interference { .. } | PlannedDesign::OutcomeAdaptive { .. } => false,
        }
    }

    fn minimum_units(&self) -> Result<u64, PlanningRefusal> {
        (1..=MINIMUM_SCAN).find(|&units| self.feasible(units)).ok_or_else(|| {
            PlanningRefusal::invalid(
                "planning.size",
                format!("no size up to {MINIMUM_SCAN} gives every arm two analysed units"),
            )
        })
    }

    fn look_units(units: u64, fractions: &[f64]) -> Vec<u64> {
        let last = fractions.len().saturating_sub(1);
        fractions
            .iter()
            .enumerate()
            .map(|(k, f)| if k == last { units } else { round_u64(f * units as f64) })
            .collect()
    }

    /// Law of a single-contrast design at `units` and `effect`.
    #[allow(clippy::too_many_lines, reason = "one arm per design")]
    fn contrast_law(&self, units: u64, effect: f64) -> ContrastLaw {
        let kept = self.kept();
        let (vc, vt) = self.arm_variances(effect);
        let covariates = f64::from(self.request.assumptions.covariates);
        match &self.design {
            PlannedDesign::Complete | PlannedDesign::GroupSequential { .. } => {
                let (c, t) = self.two_arm(units);
                let (nc, nt) = (c as f64 * kept, t as f64 * kept);
                ContrastLaw::single(vc / nc + vt / nt, self.df(nc + nt - 2.0 - covariates), None)
            }
            PlannedDesign::Bernoulli => {
                let p = self.treatment_probability();
                let n = units;
                let component = |treated: u64, weight: f64| {
                    let (nc, nt) = ((n - treated) as f64 * kept, treated as f64 * kept);
                    let variance =
                        if nc >= 2.0 && nt >= 2.0 { vc / nc + vt / nt } else { f64::INFINITY };
                    Component { weight, variance, df: self.df(nc + nt - 2.0 - covariates) }
                };
                if n > BERNOULLI_MIXTURE_LIMIT {
                    return ContrastLaw {
                        components: vec![component(round_u64(n as f64 * p), 1.0)],
                        design_effect: None,
                    };
                }
                let mean = n as f64 * p;
                let sd = (n as f64 * p * (1.0 - p)).sqrt();
                let low = floor_u64((mean - 10.0 * sd - 1.0).max(0.0));
                let high = (floor_u64(mean + 10.0 * sd + 2.0)).min(n);
                let components =
                    (low..=high).map(|k| component(k, binomial_pmf(n, k, p))).collect();
                ContrastLaw { components, design_effect: None }
            }
            PlannedDesign::Stratified { strata } => {
                let sizes = allocate(units, &strata.iter().map(|s| s.share).collect::<Vec<_>>());
                let total: f64 = sizes.iter().map(|s| *s as f64).sum();
                let mut variance = 0.0;
                for (stratum, size) in strata.iter().zip(&sizes) {
                    let (c, t) = self.two_arm(*size);
                    let (nc, nt) = (c as f64 * kept, t as f64 * kept);
                    let (sc, st) = match self.request.assumptions.outcome {
                        OutcomeModel::Continuous { .. } => {
                            let v = stratum.within * stratum.within;
                            (v, v)
                        }
                        OutcomeModel::Binary { .. } => {
                            let (p0, p1) = (rate(stratum.within), rate(stratum.within + effect));
                            (p0 * (1.0 - p0), p1 * (1.0 - p1))
                        }
                    };
                    let w = *size as f64 / total;
                    variance += w * w * (sc / nc + st / nt);
                }
                let df = total * kept - 2.0 * strata.len() as f64;
                ContrastLaw::single(variance, self.df(df), None)
            }
            PlannedDesign::Cluster { mean_cluster_size, icc, cluster_size_cv }
            | PlannedDesign::Interference {
                exposure:
                    Some(super::request::InterferenceExposure::ContainedInClusters {
                        mean_cluster_size,
                        icc,
                        cluster_size_cv,
                    }),
            } => {
                let (kc, kt) = self.two_arm(units);
                let m = mean_cluster_size * kept;
                let cv2 = cluster_size_cv * cluster_size_cv;
                let design_effect = 1.0 + ((1.0 + cv2) * m - 1.0) * icc;
                let (mc, mt) = (self.control_level(), self.control_level() + effect);
                let arm = |level: f64, variance: f64, clusters: u64| {
                    (level * level * cv2 + variance * design_effect / m) / clusters as f64
                };
                let variance = arm(mc, vc, kc) + arm(mt, vt, kt);
                ContrastLaw::single(variance, self.df((kc + kt) as f64 - 2.0), Some(design_effect))
            }
            PlannedDesign::Factorial { contrast } => {
                let cells = allocate(units, &[1.0; 4]);
                let treated_cells: &[usize] = match contrast {
                    FactorialContrast::MainEffect => &[1, 3],
                    FactorialContrast::Interaction => &[3],
                };
                let mut sum = 0.0;
                for (index, n) in cells.iter().enumerate() {
                    let v = if treated_cells.contains(&index) { vt } else { vc };
                    sum += v / (*n as f64 * kept);
                }
                let variance = match contrast {
                    FactorialContrast::MainEffect => 0.25 * sum,
                    FactorialContrast::Interaction => sum,
                };
                ContrastLaw::single(variance, self.df(units as f64 * kept - 4.0), None)
            }
            PlannedDesign::Switchback {
                sequences,
                periods_per_block,
                autocorrelation,
                washout_periods,
                ..
            } => {
                let analysed = periods_per_block - washout_periods;
                let l = analysed as f64;
                let p = self.treatment_probability();
                let mu = self.control_level();
                let noise = vc * l * block_inflation(analysed, *autocorrelation);
                let block = (l * l * (mu + effect).powi(2) + noise) / p
                    + (l * l * mu * mu + noise) / (1.0 - p)
                    - l * l * effect * effect;
                let blocks = (sequences * units) as f64;
                let inflation = block_inflation(analysed, *autocorrelation);
                ContrastLaw::single(
                    block / (blocks * l * l),
                    Some(*sequences as f64 - 1.0),
                    Some(inflation),
                )
            }
            PlannedDesign::MultiArm { .. }
            | PlannedDesign::Interference { exposure: None }
            | PlannedDesign::OutcomeAdaptive { .. } => {
                ContrastLaw::single(f64::INFINITY, None, None)
            }
        }
    }

    fn single_contrast_power(&self, units: u64, effect: f64) -> PowerDetail {
        let law = self.contrast_law(units, effect);
        let (alpha, two_sided) = (self.alpha(), self.two_sided());
        let mut power = 0.0;
        for c in &law.components {
            if c.variance.is_finite() && c.variance > 0.0 {
                power +=
                    c.weight * t_test_power(effect / c.variance.sqrt(), c.df, alpha, two_sided);
            }
        }
        let variance = law.mean_variance();
        let df = law.reference_df();
        let method = match (&self.design, self.binary()) {
            (PlannedDesign::Bernoulli, _) => "binomial_mixture_of_test_power",
            (_, true) => "normal_wald",
            _ => "noncentral_t",
        };
        PowerDetail {
            power: power.clamp(0.0, 1.0),
            variance,
            noncentrality: Some(effect / variance.sqrt()),
            df,
            critical: Some(critical_value(alpha, df, two_sided)),
            design_effect: law.design_effect,
            family: None,
            sequential: None,
            method,
        }
    }

    fn dunnett_critical(&self, comparisons: EquicorrelatedComparisons) -> f64 {
        let key = (comparisons.correlation.to_bits(), comparisons.df.unwrap_or(-1.0).to_bits());
        if let Some(value) = self.dunnett.borrow().get(&key) {
            return *value;
        }
        let value = comparisons.dunnett_critical(self.alpha()).unwrap_or(f64::NAN);
        self.dunnett.borrow_mut().insert(key, value);
        value
    }

    fn multi_arm_power(
        &self,
        units: u64,
        effect: f64,
        treatment_arms: usize,
        comparison: ArmComparison,
    ) -> PowerDetail {
        let kept = self.kept();
        let arms = self.arm_units(units);
        let (vc, vt) = self.arm_variances(effect);
        let analysed: Vec<f64> = arms.iter().map(|n| *n as f64 * kept).collect();
        let total: f64 = analysed.iter().sum();
        let df = self.df(total - (treatment_arms + 1) as f64);
        let smallest = analysed[1..].iter().copied().fold(f64::INFINITY, f64::min);
        let control_part = vc / analysed[0];
        let variance = control_part + vt / smallest;
        match comparison {
            ArmComparison::Omnibus => {
                let mean = analysed[1..].iter().sum::<f64>() * effect / total;
                let mut lambda = analysed[0] * mean * mean;
                for n in &analysed[1..] {
                    lambda += n * (effect - mean).powi(2);
                }
                lambda /= vc;
                let df1 = treatment_arms as f64;
                let df2 = total - df1 - 1.0;
                PowerDetail {
                    power: f_test_power(lambda, df1, df2, self.alpha()),
                    variance,
                    noncentrality: Some(lambda),
                    df: Some(df2),
                    critical: Some(antecedent_stats::power::f_critical_value(
                        self.alpha(),
                        df1,
                        df2,
                    )),
                    design_effect: None,
                    family: None,
                    sequential: None,
                    method: "noncentral_f",
                }
            }
            ArmComparison::ManyToOne { multiplicity, power } => {
                let comparisons = EquicorrelatedComparisons {
                    comparisons: treatment_arms,
                    correlation: (control_part / variance).clamp(0.0, 1.0 - 1e-12),
                    df,
                    two_sided: self.two_sided(),
                };
                let k = treatment_arms;
                let critical: Vec<f64> = match multiplicity {
                    Multiplicity::Bonferroni => {
                        vec![critical_value(self.alpha() / k as f64, df, self.two_sided()); k]
                    }
                    Multiplicity::Holm => (0..k)
                        .map(|j| {
                            critical_value(self.alpha() / (k - j) as f64, df, self.two_sided())
                        })
                        .collect(),
                    Multiplicity::Dunnett => vec![self.dunnett_critical(comparisons); k],
                };
                let delta = effect / variance.sqrt();
                let tail = comparisons
                    .rejection_tail(delta, &critical)
                    .unwrap_or_else(|| vec![f64::NAN; k]);
                let family = FamilyCharacteristics {
                    any: tail[0],
                    all: tail[k - 1],
                    per_comparison: tail.iter().sum::<f64>() / k as f64,
                    critical: critical.clone(),
                    correlation: comparisons.correlation,
                };
                let value = match power {
                    FamilyPower::PerComparison => family.per_comparison,
                    FamilyPower::Any => family.any,
                    FamilyPower::All => family.all,
                };
                PowerDetail {
                    power: value,
                    variance,
                    noncentrality: Some(delta),
                    df,
                    critical: critical.first().copied(),
                    design_effect: None,
                    family: Some(family),
                    sequential: None,
                    method: "equicorrelated_comparisons",
                }
            }
        }
    }

    fn boundary(&self, looks: &[u64], units: u64) -> Result<Vec<f64>, PlanningRefusal> {
        if let Some(found) = self.boundaries.borrow().get(looks) {
            return Ok(found.clone());
        }
        let fractions: Vec<f64> = looks.iter().map(|n| *n as f64 / units as f64).collect();
        let PlannedDesign::GroupSequential { spending, .. } = &self.design else {
            return Err(PlanningRefusal::invalid("planning.group_sequential", "not sequential"));
        };
        let critical = boundaries(&fractions, self.alpha(), *spending, self.sidedness(), &|| false)
            .map_err(|error| {
                PlanningRefusal::invalid("planning.group_sequential", error.to_string())
            })?
            .critical;
        self.boundaries.borrow_mut().insert(looks.to_vec(), critical.clone());
        Ok(critical)
    }

    fn sidedness(&self) -> Sidedness {
        if self.two_sided() { Sidedness::TwoSidedSymmetric } else { Sidedness::OneSidedUpper }
    }

    fn sequential_power(
        &self,
        units: u64,
        effect: f64,
        fractions: &[f64],
    ) -> Result<PowerDetail, PlanningRefusal> {
        let looks = Self::look_units(units, fractions);
        let critical = self.boundary(&looks, units)?;
        let law = self.contrast_law(units, effect);
        let variance = law.mean_variance();
        let drift = effect / variance.sqrt();
        let actual: Vec<f64> = looks.iter().map(|n| *n as f64 / units as f64).collect();
        let crossing = |drift: f64| {
            crossing_probabilities(&actual, &critical, self.sidedness(), drift, &|| false).map_err(
                |error| PlanningRefusal::invalid("planning.group_sequential", error.to_string()),
            )
        };
        let alternative = crossing(drift)?;
        let null = crossing(0.0)?;
        let expected = |stops: &[f64]| {
            let stopped: f64 = stops.iter().sum();
            stops.iter().zip(&looks).map(|(p, n)| p * *n as f64).sum::<f64>()
                + (1.0 - stopped).max(0.0) * units as f64
        };
        let sequential = SequentialCharacteristics {
            critical: critical.clone(),
            look_units: looks.clone(),
            expected_units_alternative: expected(&alternative),
            expected_units_null: expected(&null),
            stop_probability_alternative: alternative.clone(),
            stop_probability_null: null,
            fixed_design_units: None,
        };
        Ok(PowerDetail {
            power: alternative.iter().sum::<f64>().clamp(0.0, 1.0),
            variance,
            noncentrality: Some(drift),
            df: None,
            critical: critical.last().copied(),
            design_effect: None,
            family: None,
            sequential: Some(sequential),
            method: "recursive_integration",
        })
    }

    /// Frequentist power of the planned test at `units` and `effect`.
    fn power(&self, units: u64, effect: f64) -> Result<PowerDetail, PlanningRefusal> {
        match &self.design {
            PlannedDesign::MultiArm { treatment_arms, comparison } => {
                Ok(self.multi_arm_power(units, effect, *treatment_arms, *comparison))
            }
            PlannedDesign::GroupSequential { information_fractions, .. } => {
                self.sequential_power(units, effect, information_fractions)
            }
            _ => Ok(self.single_contrast_power(units, effect)),
        }
    }

    /// Nodes `(effect, weight)` of a normal prior.
    fn prior_nodes(mean: f64, sd: f64) -> Vec<(f64, f64)> {
        if sd == 0.0 {
            return vec![(mean, 1.0)];
        }
        let (nodes, weights) = gauss_hermite_standard_normal(PRIOR_NODES);
        nodes.into_iter().zip(weights).map(|(z, w)| (mean + sd * z, w)).collect()
    }

    /// Bound a prior draw to the effects the outcome law admits.
    fn admissible_effect(&self, effect: f64) -> f64 {
        match self.request.assumptions.outcome {
            OutcomeModel::Binary { control_rate } => {
                effect.clamp(RATE_GUARD - control_rate, 1.0 - RATE_GUARD - control_rate)
            }
            OutcomeModel::Continuous { .. } => effect,
        }
    }

    /// The criterion at `units`, with the detail of the planned test.
    fn criterion(
        &self,
        units: u64,
        effect: Option<f64>,
    ) -> Result<(f64, PowerDetail), PlanningRefusal> {
        match self.request.criterion {
            PlanningCriterion::Power { .. } => {
                let detail = self.power(units, effect.unwrap_or(0.0))?;
                Ok((detail.power, detail))
            }
            PlanningCriterion::Assurance { design_prior, success, .. } => {
                let mut detail = self.power(units, self.admissible_effect(design_prior.mean))?;
                let value = match success {
                    SuccessCriterion::Significance => {
                        let mut total = 0.0;
                        for (theta, w) in Self::prior_nodes(design_prior.mean, design_prior.sd) {
                            total += w * self.power(units, self.admissible_effect(theta))?.power;
                        }
                        detail.method = "prior_averaged_power";
                        total
                    }
                    SuccessCriterion::Posterior { analysis_prior, threshold, probability } => {
                        let law =
                            self.contrast_law(units, self.admissible_effect(design_prior.mean));
                        let z = norm_inv(probability);
                        let prior_precision = 1.0 / (analysis_prior.sd * analysis_prior.sd);
                        let mut total = 0.0;
                        for c in &law.components {
                            let value = if c.variance.is_finite() {
                                let posterior_var = 1.0 / (prior_precision + 1.0 / c.variance);
                                let sd = posterior_var.sqrt();
                                let cut = c.variance
                                    * ((threshold + z * sd) / posterior_var
                                        - analysis_prior.mean * prior_precision);
                                let spread = (design_prior.sd.powi(2) + c.variance).sqrt();
                                norm_cdf((design_prior.mean - cut) / spread)
                            } else {
                                let prior_only =
                                    norm_cdf((analysis_prior.mean - threshold) / analysis_prior.sd);
                                if prior_only >= probability { 1.0 } else { 0.0 }
                            };
                            total += c.weight * value;
                        }
                        detail.method = "conjugate_normal";
                        total
                    }
                };
                detail.power = detail.power.clamp(0.0, 1.0);
                Ok((value.clamp(0.0, 1.0), detail))
            }
            PlanningCriterion::PosteriorPrecision { analysis_prior, .. } => {
                let law = self.contrast_law(units, self.admissible_effect(analysis_prior.mean));
                let prior_precision = 1.0 / (analysis_prior.sd * analysis_prior.sd);
                let mut total = 0.0;
                for c in &law.components {
                    let data_precision =
                        if c.variance.is_finite() { 1.0 / c.variance } else { 0.0 };
                    total += c.weight / (prior_precision + data_precision).sqrt();
                }
                let mut detail = self.single_contrast_power(units, 0.0);
                detail.method = "conjugate_normal";
                Ok((total, detail))
            }
            PlanningCriterion::CorrectDecision {
                design_prior,
                analysis_prior,
                threshold,
                probability,
                ..
            } => {
                let law = self.contrast_law(units, self.admissible_effect(design_prior.mean));
                let z = norm_inv(probability);
                let prior_precision = 1.0 / (analysis_prior.sd * analysis_prior.sd);
                let mut total = 0.0;
                for c in &law.components {
                    total += c.weight
                        * correct_decision(
                            c.variance,
                            prior_precision,
                            analysis_prior.mean,
                            analysis_prior.sd,
                            design_prior.mean,
                            design_prior.sd,
                            threshold,
                            probability,
                            z,
                        );
                }
                let mut detail = self.power(units, self.admissible_effect(design_prior.mean))?;
                detail.method = "conjugate_normal";
                Ok((total.clamp(0.0, 1.0), detail))
            }
        }
    }

    fn meets(&self, value: f64) -> bool {
        let target = self.request.criterion.target();
        if self.request.criterion.larger_is_better() { value >= target } else { value <= target }
    }

    /// Smallest feasible size whose criterion meets its target.
    fn solve_units(&self, minimum: u64, max_units: u64) -> Result<u64, PlanningRefusal> {
        let effect = self.request.assumptions.effect;
        let at = |units: u64| -> Result<bool, PlanningRefusal> {
            Ok(self.feasible(units) && self.meets(self.criterion(units, effect)?.0))
        };
        if minimum > max_units {
            return Err(PlanningRefusal::new(
                antecedent_core::reason_code!("evidence_cannot_reach_level"),
                "planning.size",
                format!("the smallest feasible size {minimum} exceeds max_units {max_units}"),
                Some(RELAX_TARGET),
            ));
        }
        if at(minimum)? {
            return Ok(minimum);
        }
        let mut high = minimum;
        let mut low = loop {
            let next = high.saturating_mul(2).min(max_units);
            if next == high {
                let reached = self.criterion(max_units, effect)?.0;
                return Err(PlanningRefusal::new(
                    antecedent_core::reason_code!("evidence_cannot_reach_level"),
                    "planning.size",
                    format!(
                        "the {} reaches {reached:.6} at max_units {max_units}, short of the \
                         target {}{}",
                        self.request.criterion.as_str(),
                        self.request.criterion.target(),
                        self.limit_note()
                    ),
                    Some(RELAX_TARGET),
                ));
            }
            let previous = high;
            high = next;
            if at(high)? {
                break previous;
            }
        };
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            if at(mid)? {
                high = mid;
            } else {
                low = mid;
            }
        }
        Ok(high)
    }

    /// The value the criterion approaches as the size grows, when it is bounded below one.
    fn limit_note(&self) -> String {
        match self.request.criterion {
            PlanningCriterion::Assurance { design_prior, success, .. } => {
                let threshold = match success {
                    SuccessCriterion::Significance => 0.0,
                    SuccessCriterion::Posterior { threshold, .. } => threshold,
                };
                let limit = if design_prior.sd > 0.0 {
                    let one = norm_sf((threshold - design_prior.mean) / design_prior.sd);
                    if matches!(success, SuccessCriterion::Significance) && self.two_sided() {
                        1.0
                    } else {
                        one
                    }
                } else if design_prior.mean > threshold {
                    1.0
                } else {
                    0.0
                };
                format!(
                    "; as the size grows the assurance approaches {limit:.6}, the design-prior \
                     probability of an effect beyond the success threshold"
                )
            }
            _ => String::new(),
        }
    }

    /// Smallest positive effect the planned test detects with the target power.
    fn solve_effect(&self, units: u64) -> Result<f64, PlanningRefusal> {
        let target = self.request.criterion.target();
        let upper_bound = match self.request.assumptions.outcome {
            OutcomeModel::Binary { control_rate } => {
                let mut ceiling = 1.0 - control_rate;
                if let PlannedDesign::Stratified { strata } = &self.design {
                    for stratum in strata {
                        ceiling = ceiling.min(1.0 - stratum.within);
                    }
                }
                Some(ceiling - 1e-6)
            }
            OutcomeModel::Continuous { .. } => None,
        };
        let power =
            |effect: f64| -> Result<f64, PlanningRefusal> { Ok(self.power(units, effect)?.power) };
        let mut high = match upper_bound {
            Some(bound) => bound,
            None => match self.request.assumptions.outcome {
                OutcomeModel::Continuous { sd, .. } => sd,
                OutcomeModel::Binary { .. } => 0.5,
            },
        };
        let mut reached = power(high)?;
        while reached < target {
            if upper_bound.is_some() || high > 1e6 * self.scale() {
                return Err(PlanningRefusal::new(
                    antecedent_core::reason_code!("evidence_cannot_reach_level"),
                    "planning.effect",
                    format!(
                        "no admissible effect reaches power {target} at {units} units (power \
                         {reached:.6} at effect {high})"
                    ),
                    Some(RELAX_TARGET),
                ));
            }
            high *= 2.0;
            reached = power(high)?;
        }
        let mut low = 0.0;
        for _ in 0..EFFECT_BISECTIONS {
            let mid = 0.5 * (low + high);
            if power(mid)? >= target {
                high = mid;
            } else {
                low = mid;
            }
            if high - low <= 1e-12 * high.max(1e-12) {
                break;
            }
        }
        Ok(high)
    }

    fn scale(&self) -> f64 {
        match self.request.assumptions.outcome {
            OutcomeModel::Continuous { sd, .. } => sd,
            OutcomeModel::Binary { .. } => 1.0,
        }
    }

    /// Units a fixed two-arm design with the z test needs for the target power.
    fn fixed_design_units(&self, effect: f64) -> u64 {
        let (vc, vt) = self.arm_variances(effect);
        let r = self.ratio();
        let per_unit = (vc * (1.0 + r) + vt * (1.0 + r) / r) / self.kept();
        let z_alpha =
            norm_inv(1.0 - if self.two_sided() { self.alpha() / 2.0 } else { self.alpha() });
        let z_beta = norm_inv(self.request.criterion.target());
        floor_u64((per_unit * ((z_alpha + z_beta) / effect).powi(2)).ceil())
    }

    fn assumption_labels(&self) -> Vec<String> {
        let a = &self.request.assumptions;
        let mut labels = vec![format!("outcome={}", a.outcome.as_str())];
        match a.outcome {
            OutcomeModel::Continuous { sd, control_mean } => {
                labels.push(format!("outcome_sd={sd}"));
                if let Some(mean) = control_mean {
                    labels.push(format!("control_mean={mean}"));
                }
            }
            OutcomeModel::Binary { control_rate } => {
                labels.push(format!("control_rate={control_rate}"));
            }
        }
        if let Some(effect) = a.effect {
            labels.push(format!("effect={effect}"));
        }
        labels.push(format!("alpha={}", a.alpha));
        labels.push(format!("sidedness={}", a.sidedness.as_str()));
        labels.push(format!("allocation_ratio={}", a.allocation_ratio));
        labels.push(format!("attrition={}", a.attrition));
        if a.covariates > 0 {
            labels.push(format!("covariate_r_squared={}", a.covariate_r_squared));
            labels.push(format!("covariates={}", a.covariates));
        }
        match &self.design {
            PlannedDesign::Cluster { mean_cluster_size, icc, cluster_size_cv } => {
                labels.push(format!("mean_cluster_size={mean_cluster_size}"));
                labels.push(format!("icc={icc}"));
                labels.push(format!("cluster_size_cv={cluster_size_cv}"));
            }
            PlannedDesign::Switchback {
                sequences,
                periods_per_block,
                autocorrelation,
                carryover,
                washout_periods,
            } => {
                labels.push(format!("sequences={sequences}"));
                labels.push(format!("periods_per_block={periods_per_block}"));
                labels.push(format!("autocorrelation={autocorrelation}"));
                labels.push(format!("carryover={carryover}"));
                labels.push(format!("washout_periods={washout_periods}"));
            }
            PlannedDesign::GroupSequential { information_fractions, spending } => {
                labels.push(format!("information_fractions={information_fractions:?}"));
                labels.push(format!("spending={}", spending.as_str()));
            }
            _ => {}
        }
        match self.request.criterion {
            PlanningCriterion::Assurance { design_prior, .. }
            | PlanningCriterion::CorrectDecision { design_prior, .. } => {
                labels.push(format!(
                    "design_prior=normal({}, {})",
                    design_prior.mean, design_prior.sd
                ));
            }
            _ => {}
        }
        labels
    }
}

/// Probability that the posterior decision is right, for one contrast variance.
#[allow(clippy::too_many_arguments, reason = "the closed form reads every declared number")]
fn correct_decision(
    variance: f64,
    prior_precision: f64,
    analysis_mean: f64,
    analysis_sd: f64,
    design_mean: f64,
    design_sd: f64,
    threshold: f64,
    probability: f64,
    z: f64,
) -> f64 {
    if !variance.is_finite() {
        let declare = norm_cdf((analysis_mean - threshold) / analysis_sd) >= probability;
        let above = if design_sd > 0.0 {
            norm_sf((threshold - design_mean) / design_sd)
        } else if design_mean > threshold {
            1.0
        } else {
            0.0
        };
        return if declare { above } else { 1.0 - above };
    }
    let posterior_var = 1.0 / (prior_precision + 1.0 / variance);
    let cut = variance
        * ((threshold + z * posterior_var.sqrt()) / posterior_var
            - analysis_mean * prior_precision);
    let sd = variance.sqrt();
    let correct_at = |theta: f64| {
        if theta > threshold { norm_sf((cut - theta) / sd) } else { norm_cdf((cut - theta) / sd) }
    };
    if design_sd == 0.0 {
        return correct_at(design_mean);
    }
    let (nodes, weights) = gauss_legendre(8);
    let split_z = (threshold - design_mean) / design_sd;
    let mut total = 0.0;
    for (low, high) in [(-10.0_f64, split_z.clamp(-10.0, 10.0)), (split_z.clamp(-10.0, 10.0), 10.0)]
    {
        if high <= low {
            continue;
        }
        let panels = 16;
        let width = (high - low) / f64::from(panels);
        for panel in 0..panels {
            let start = low + width * f64::from(panel);
            for (u, w) in nodes.iter().zip(&weights) {
                let zz = start + 0.5 * width * (u + 1.0);
                let theta = design_mean + design_sd * zz;
                total += 0.5 * width * w * antecedent_kernels::norm_pdf(zz) * correct_at(theta);
            }
        }
    }
    total
}

/// Plan `request` analytically: validate, solve in the requested direction and
/// report the operating characteristics at the planned size.
///
/// # Errors
///
/// A [`PlanningRefusal`]: an invalid or unplannable request, a target the
/// design does not reach within the declared bound, or a size below the
/// design's smallest feasible size.
pub fn plan_analytic(request: &PlanningRequest) -> Result<PlanningResult, PlanningRefusal> {
    request.validate()?;
    let planner = Planner::new(request);
    let minimum = planner.minimum_units()?;
    let (units, effect) = match request.goal {
        PlanningGoal::SampleSize { max_units } => {
            (planner.solve_units(minimum, max_units)?, request.assumptions.effect)
        }
        PlanningGoal::Evaluate { units } | PlanningGoal::MinimumDetectableEffect { units }
            if !planner.feasible(units) =>
        {
            return Err(PlanningRefusal::invalid(
                "planning.size",
                format!(
                    "{units} {}s leave an arm with fewer than two analysed units; the smallest \
                     feasible size is {minimum}",
                    request.design.unit()
                ),
            ));
        }
        PlanningGoal::Evaluate { units } => (units, request.assumptions.effect),
        PlanningGoal::MinimumDetectableEffect { units } => {
            (units, Some(planner.solve_effect(units)?))
        }
    };
    let (value, detail) = planner.criterion(units, effect)?;
    let mut sequential = detail.sequential;
    if let (Some(seq), PlanningGoal::SampleSize { .. }, Some(effect)) =
        (sequential.as_mut(), request.goal, effect)
    {
        if matches!(request.criterion, PlanningCriterion::Power { .. }) && effect != 0.0 {
            seq.fixed_design_units = Some(planner.fixed_design_units(effect));
        }
    }
    let power = (!matches!(request.criterion, PlanningCriterion::PosteriorPrecision { .. }))
        .then_some(detail.power);
    Ok(PlanningResult {
        request_identity: request.identity(),
        design: request.design.as_str(),
        analysis: request.design.analysis(&request.assumptions),
        criterion: request.criterion.as_str(),
        goal: request.goal.as_str(),
        unit: request.design.unit(),
        units,
        arm_units: planner.arm_units(units),
        enrolled_individuals: planner.enrolled_individuals(units),
        analysed_individuals: planner.analysed_individuals(units),
        effect,
        target: request.criterion.target(),
        target_met: planner.meets(value),
        analytic: AnalyticCharacteristics {
            value,
            estimator_variance: detail.variance,
            noncentrality: detail.noncentrality,
            reference_df: detail.df,
            critical_value: detail.critical,
            design_effect: detail.design_effect,
            power,
            family: detail.family,
            sequential,
            method: detail.method,
        },
        simulation: None,
        assumptions: planner.assumption_labels(),
        calibration: PLANNING_CALIBRATION,
    })
}
