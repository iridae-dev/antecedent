//! Derived uncertainty of an adaptive procedure whose units contribute
//! several correlated measurements: repeated measures within a unit, or
//! members within a cluster with a unit-level random effect.
//!
//! The Bernoulli procedures of [`super::procedure_law`] observe one outcome
//! per unit. A unit that is measured `m` times (or a cluster of `m` members)
//! contributes outcomes that are not independent, so neither the exact
//! binomial-tree law nor an analysis that treats the measurements as
//! independent applies. This module declares the dependence and derives the
//! uncertainty of the *whole procedure* under it:
//!
//! * The within-unit dependence is declared, never defaulted
//!   ([`OutcomeStructure`]): a compound-symmetric or first-order
//!   autoregressive correlation of repeated measures with a stated variance,
//!   or a unit-level random effect with a stated between-unit and within-unit
//!   variance. A missing element is refused with its own detail.
//! * Assignment is at the unit level. The policy is evaluated from the
//!   *unit-level summaries* of units whose follow-up has completed
//!   (`lag_looks` batches after enrolment) and nothing else; there is no
//!   interim analysis and no stopping rule, so the monitoring strategy of the
//!   cell is `final_analysis_only` and the stopping kind `no_stopping`.
//!   Other monitoring and stopping combinations are separate cells and are
//!   refused here.
//! * The final analysis is declared ([`FinalAnalysis`]): the contrast of arm
//!   means of unit-level summaries with a cluster-robust (unit-level)
//!   variance, the same with generalized-least-squares unit summaries, a
//!   pooled analysis that treats the measurements as independent (admitted
//!   only when independence is declared), or the adaptively weighted
//!   augmented score of [`super::weighting`] over unit summaries (the
//!   weighting route).
//! * Exact closed forms are the oracle: the variance of a unit's mean
//!   measurement is `variance * (1 + (m - 1) * rho) / m` under compound
//!   symmetry (the design effect is `1 + (m - 1) * rho`), and for a
//!   fixed-allocation design the contrast law is an exactly known mixture of
//!   normals ([`exact_fixed_contrast_law`]). The adaptive procedure's law has
//!   no closed form, so it is simulated end to end under a declared seed
//!   ([`simulate_repeated_measures`]) and the Monte Carlo error is reported
//!   separately from the exact quantities.
//!
//! Premise gates are per paradigm: a fixed-sample analysis after
//! response-adaptive assignment is refused, an independence assumption for
//! dependent measurements is refused, and a guarantee derived for one cell is
//! never carried to another ([`CellLabel::transfer_to`]).
//!
//! Every result is labelled [`DERIVED_UNMEASURED`]; no coverage is claimed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "the correlation matrix and weight vectors are addressed by unit-measurement index"
)]

use std::collections::VecDeque;

use antecedent_core::CausalRng;
use antecedent_kernels::norm_cdf;

use super::AdaptiveInferenceError as Error;
use super::procedure_contract::{
    ContractGap, MAX_PROCEDURE_ARMS, MAX_PROCEDURE_UNITS, ProcedureContractError, StoppingRule,
};
use super::procedure_simulation::MAX_SIMULATION_REPLICATIONS;
use super::procedure_uncertainty::{DERIVED_UNMEASURED, UncertaintyParadigm};
use super::weighting::{
    Estimand, OutcomeModelPrior, WeightScheme, adaptive_scores, weighted_estimate,
};

/// Most measurements per unit (or members per cluster).
pub const MAX_MEASUREMENTS_PER_UNIT: u32 = 32;
/// Most policy-update batches of one procedure.
pub const MAX_UNIT_BATCHES: usize = 400;
/// Longest follow-up delay, in batches.
pub const MAX_UNIT_LAG: usize = 32;

const STREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const TIE_TOLERANCE: f64 = 1e-12;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

const REPEATED_MEASURES: &str = "repeated_measures";
const HIERARCHICAL: &str = "hierarchical_clustered";

/// Registered reason code of every declaration refusal for an element that
/// is not stated, by the detail it carries.
#[must_use]
pub fn incomplete_reason_code() -> &'static str {
    antecedent_core::reason_code!("adaptive_contract_incomplete")
}

/// Registered reason code of an invalid declared element
/// (`measurements_invalid`, `variance_invalid`, `correlation_invalid`,
/// `analysis_invalid`, `repeated_measures_stopping_not_derived`).
#[must_use]
pub fn invalid_reason_code() -> &'static str {
    antecedent_core::reason_code!("invalid_argument")
}

/// Registered reason code of every premise refusal of this module
/// (`analysis_route_mismatch`, `exact_law_unavailable`,
/// `guarantee_not_transferable`, `independence_assumed_for_dependent_measurements`,
/// `repeated_measures_paradigm_not_derived`).
#[must_use]
pub fn premise_reason_code() -> &'static str {
    antecedent_core::reason_code!("cell_not_licensed")
}

// -- Declaration ----------------------------------------------------------------------------------

/// Correlation of two measurements of one unit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum WithinUnitCorrelation {
    /// Every pair of measurements has correlation `rho`; `rho` lies in
    /// `(-1 / (m - 1), 1)`.
    CompoundSymmetric {
        /// The common correlation.
        rho: f64,
    },
    /// First-order autoregressive: measurements `j` and `k` have correlation
    /// `rho^|j - k|`, `|rho| < 1`.
    Autoregressive {
        /// The lag-one correlation.
        rho: f64,
    },
}

/// The outcome structure of a unit. Each declared quantity is an `Option` so
/// that a missing declaration is representable and refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OutcomeStructure {
    /// `measurements` ordered measurements of one unit with a common marginal
    /// variance and a declared within-unit correlation.
    RepeatedMeasures {
        /// Measurements per unit, between 2 and [`MAX_MEASUREMENTS_PER_UNIT`].
        measurements: u32,
        /// Marginal variance of one measurement; positive.
        variance: f64,
        /// The within-unit correlation; `None` is refused.
        correlation: Option<WithinUnitCorrelation>,
    },
    /// A cluster of `members` units-within-cluster with a unit-level random
    /// effect: `y_ij = mu + b_i + e_ij`, `b_i ~ N(0, between)`,
    /// `e_ij ~ N(0, within)`.
    Hierarchical {
        /// Members per cluster, between 2 and [`MAX_MEASUREMENTS_PER_UNIT`].
        members: u32,
        /// Between-unit variance of the random effect; non-negative. `None`
        /// is refused.
        between_variance: Option<f64>,
        /// Within-unit variance; positive. `None` is refused.
        within_variance: Option<f64>,
    },
}

/// Unit-level assignment rule, evaluated from the unit-level summaries of the
/// units whose follow-up has completed.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum UnitAssignment {
    /// Probabilities fixed in advance.
    Fixed {
        /// One probability per arm.
        probabilities: Vec<f64>,
    },
    /// Epsilon-greedy over prior-shrunk running means of the unit summaries.
    /// Arms tied for the highest mean share the greedy mass equally.
    EpsilonGreedy {
        /// Exploration probability in `[0, 1]`.
        epsilon: f64,
        /// The running mean of an arm is shrunk to `prior.mean` with
        /// `prior.strength` pseudo-units.
        prior: OutcomeModelPrior,
    },
}

impl UnitAssignment {
    /// Whether the rule depends on the observed history.
    #[must_use]
    pub const fn is_adaptive(&self) -> bool {
        matches!(self, Self::EpsilonGreedy { .. })
    }
}

/// How the measurements of a unit are reduced to one unit-level value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnitSummary {
    /// The unweighted mean of the measurements.
    Mean,
    /// The generalized-least-squares estimate of the unit level under the
    /// declared correlation (equal to the mean under compound symmetry and
    /// for a random-effect cluster with equal members).
    Gls,
}

/// The declared final-analysis estimator of the contrast `plus - minus`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AnalysisEstimator {
    /// Difference of arm means of the unit summaries with the unit-level
    /// (cluster-robust, Welch) variance. A fixed-sample analysis.
    ClusterRobust {
        /// Unit reduction.
        summary: UnitSummary,
    },
    /// Difference of arm means of all measurements with the variance that
    /// treats the measurements as independent. Admitted only when the
    /// declared dependence is zero.
    PooledIndependent,
    /// Adaptively weighted augmented score over the unit summaries (the
    /// weighting route), valid under response-adaptive assignment.
    AdaptivelyWeighted {
        /// Unit reduction.
        summary: UnitSummary,
        /// Score weights.
        scheme: WeightScheme,
        /// Prior of the predictable outcome model.
        prior: OutcomeModelPrior,
        /// Smallest assignment probability an estimand arm may have had.
        probability_floor: f64,
    },
}

/// The estimator and its contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalAnalysis {
    /// The estimator.
    pub estimator: AnalysisEstimator,
    /// Arm entering with a plus sign.
    pub plus: usize,
    /// Arm entering with a minus sign.
    pub minus: usize,
}

/// Everything a producer states about the repeated-measures procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresDeclaration {
    /// Number of arms.
    pub arms: usize,
    /// The unit-level assignment rule.
    pub assignment: Option<UnitAssignment>,
    /// Units enrolled per policy-update batch. All ones is fully sequential
    /// (the epsilon-greedy cell); a batch above one is a batched design.
    pub enrolment: Option<Vec<u32>>,
    /// Batches between a unit's enrolment and its summary becoming available
    /// to the policy and the analysis.
    pub lag_looks: usize,
    /// The within-unit outcome structure.
    pub structure: Option<OutcomeStructure>,
    /// The final analysis.
    pub analysis: Option<FinalAnalysis>,
    /// The stopping rule; only no stopping is derived.
    pub stopping: Option<StoppingRule>,
}

// -- Covariance -----------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Dependence {
    CompoundSymmetric(f64),
    Autoregressive(f64),
    RandomEffect { between: f64, within: f64 },
}

#[derive(Clone, Debug, PartialEq)]
struct Covariance {
    measurements: usize,
    marginal_variance: f64,
    dependence: Dependence,
    /// Row-major `m x m` correlation matrix.
    correlation: Vec<f64>,
    /// Row-major lower Cholesky factor of the correlation matrix.
    factor: Vec<f64>,
}

fn correlation_matrix(measurements: usize, entry: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut matrix = vec![0.0; measurements * measurements];
    for i in 0..measurements {
        for j in 0..measurements {
            matrix[i * measurements + j] = if i == j { 1.0 } else { entry(i, j) };
        }
    }
    matrix
}

fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return None;
                }
                lower[i * n + i] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Some(lower)
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl Covariance {
    fn build(structure: &OutcomeStructure) -> Option<Self> {
        match structure {
            OutcomeStructure::RepeatedMeasures {
                measurements,
                variance,
                correlation: Some(correlation),
            } => {
                let m = *measurements as usize;
                let (dependence, matrix) = match *correlation {
                    WithinUnitCorrelation::CompoundSymmetric { rho } => {
                        (Dependence::CompoundSymmetric(rho), correlation_matrix(m, |_, _| rho))
                    }
                    WithinUnitCorrelation::Autoregressive { rho } => (
                        Dependence::Autoregressive(rho),
                        correlation_matrix(m, |i, j| rho.powi(i.abs_diff(j) as i32)),
                    ),
                };
                let factor = cholesky(&matrix, m)?;
                Some(Self {
                    measurements: m,
                    marginal_variance: *variance,
                    dependence,
                    correlation: matrix,
                    factor,
                })
            }
            OutcomeStructure::Hierarchical {
                members,
                between_variance: Some(between),
                within_variance: Some(within),
            } => {
                let m = *members as usize;
                let total = between + within;
                let icc = between / total;
                let matrix = correlation_matrix(m, |_, _| icc);
                let factor = cholesky(&matrix, m)?;
                Some(Self {
                    measurements: m,
                    marginal_variance: total,
                    dependence: Dependence::RandomEffect { between: *between, within: *within },
                    correlation: matrix,
                    factor,
                })
            }
            _ => None,
        }
    }

    /// Weights that reduce a unit's measurements to its summary; sum to one.
    fn weights(&self, summary: UnitSummary) -> Vec<f64> {
        let m = self.measurements;
        match (summary, &self.dependence) {
            (UnitSummary::Gls, Dependence::Autoregressive(rho)) => {
                let denominator = 2.0 + (m as f64 - 2.0) * (1.0 - rho);
                (0..m)
                    .map(|j| {
                        if j == 0 || j + 1 == m {
                            1.0 / denominator
                        } else {
                            (1.0 - rho) / denominator
                        }
                    })
                    .collect()
            }
            _ => vec![1.0 / m as f64; m],
        }
    }

    fn reduce(&self, values: &[f64], summary: UnitSummary) -> f64 {
        self.weights(summary).iter().zip(values).map(|(w, v)| w * v).sum()
    }

    /// Variance of the unit summary: `marginal * w' R w`.
    fn summary_variance(&self, summary: UnitSummary) -> f64 {
        let w = self.weights(summary);
        let m = self.measurements;
        let mut quadratic = 0.0;
        for i in 0..m {
            for j in 0..m {
                quadratic += w[i] * self.correlation[i * m + j] * w[j];
            }
        }
        self.marginal_variance * quadratic
    }

    fn draw(&self, rng: &mut CausalRng, mean: f64) -> Vec<f64> {
        let m = self.measurements;
        match &self.dependence {
            Dependence::RandomEffect { between, within } => {
                let effect = between.sqrt() * standard_normal(rng);
                let sd = within.sqrt();
                (0..m).map(|_| mean + effect + sd * standard_normal(rng)).collect()
            }
            _ => {
                let z: Vec<f64> = (0..m).map(|_| standard_normal(rng)).collect();
                let sd = self.marginal_variance.sqrt();
                (0..m)
                    .map(|i| {
                        let correlated: f64 = (0..=i).map(|k| self.factor[i * m + k] * z[k]).sum();
                        mean + sd * correlated
                    })
                    .collect()
            }
        }
    }

    fn is_independent(&self) -> bool {
        match self.dependence {
            Dependence::CompoundSymmetric(rho) | Dependence::Autoregressive(rho) => {
                rho.to_bits() << 1 == 0
            }
            Dependence::RandomEffect { between, .. } => between.to_bits() << 1 == 0,
        }
    }
}

// -- Validated spec -------------------------------------------------------------------------------

/// A validated repeated-measures or clustered procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresSpec {
    arms: usize,
    assignment: UnitAssignment,
    enrolment: Vec<u32>,
    lag_looks: usize,
    structure: OutcomeStructure,
    covariance: Covariance,
    analysis: FinalAnalysis,
}

fn probability_vector(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && (values.iter().sum::<f64>() - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE
}

fn prior_is_valid(prior: &OutcomeModelPrior) -> bool {
    prior.mean.is_finite() && prior.strength.is_finite() && prior.strength > 0.0
}

impl RepeatedMeasuresSpec {
    /// Validate a declaration.
    ///
    /// # Errors
    /// One gap per missing element (including the within-unit correlation or
    /// the between and within variances) and per invalid element; nothing is
    /// defaulted.
    #[allow(
        clippy::too_many_lines,
        reason = "one flat list of the element checks keeps each refusal next to its condition"
    )]
    pub fn new(declaration: RepeatedMeasuresDeclaration) -> Result<Self, ProcedureContractError> {
        let mut gaps = Vec::new();
        let arms = declaration.arms;
        if !(2..=MAX_PROCEDURE_ARMS).contains(&arms) {
            gaps.push(ContractGap::Invalid("adaptive_inference.arms_invalid"));
        }
        match &declaration.assignment {
            None => gaps.push(ContractGap::AssignmentPolicyMissing),
            Some(UnitAssignment::Fixed { probabilities }) => {
                if probabilities.len() != arms || !probability_vector(probabilities) {
                    gaps.push(ContractGap::Invalid("adaptive_inference.rule_mismatch"));
                }
            }
            Some(UnitAssignment::EpsilonGreedy { epsilon, prior }) => {
                if !(epsilon.is_finite() && (0.0..=1.0).contains(epsilon) && prior_is_valid(prior))
                {
                    gaps.push(ContractGap::Invalid("adaptive_inference.rule_mismatch"));
                }
            }
        }
        match &declaration.enrolment {
            None => gaps.push(ContractGap::EnrolmentScheduleMissing),
            Some(schedule) => {
                let total: u64 = schedule.iter().map(|&n| u64::from(n)).sum();
                if schedule.is_empty()
                    || schedule.len() > MAX_UNIT_BATCHES
                    || total == 0
                    || total > u64::from(MAX_PROCEDURE_UNITS)
                    || schedule[0] == 0
                {
                    gaps.push(ContractGap::Invalid("adaptive_inference.looks_not_ordered"));
                }
            }
        }
        if declaration.lag_looks > MAX_UNIT_LAG {
            gaps.push(ContractGap::Invalid("adaptive_inference.outcome_availability_invalid"));
        }
        let count_valid = |n: u32| (2..=MAX_MEASUREMENTS_PER_UNIT).contains(&n);
        match &declaration.structure {
            None => gaps.push(ContractGap::OutcomeStructureMissing),
            Some(OutcomeStructure::RepeatedMeasures { measurements, variance, correlation }) => {
                if !count_valid(*measurements) {
                    gaps.push(ContractGap::Invalid("adaptive_inference.measurements_invalid"));
                }
                if !(variance.is_finite() && *variance > 0.0) {
                    gaps.push(ContractGap::Invalid("adaptive_inference.variance_invalid"));
                }
                match correlation {
                    None => gaps.push(ContractGap::WithinUnitCorrelationMissing),
                    Some(WithinUnitCorrelation::CompoundSymmetric { rho }) => {
                        let m = f64::from(*measurements);
                        if !(rho.is_finite() && *rho < 1.0 && *rho * (m - 1.0) > -1.0) {
                            gaps.push(ContractGap::Invalid(
                                "adaptive_inference.correlation_invalid",
                            ));
                        }
                    }
                    Some(WithinUnitCorrelation::Autoregressive { rho }) => {
                        if !(rho.is_finite() && rho.abs() < 1.0) {
                            gaps.push(ContractGap::Invalid(
                                "adaptive_inference.correlation_invalid",
                            ));
                        }
                    }
                }
            }
            Some(OutcomeStructure::Hierarchical { members, between_variance, within_variance }) => {
                if !count_valid(*members) {
                    gaps.push(ContractGap::Invalid("adaptive_inference.measurements_invalid"));
                }
                match between_variance {
                    None => gaps.push(ContractGap::BetweenUnitVarianceMissing),
                    Some(v) if !(v.is_finite() && *v >= 0.0) => {
                        gaps.push(ContractGap::Invalid("adaptive_inference.variance_invalid"));
                    }
                    Some(_) => {}
                }
                match within_variance {
                    None => gaps.push(ContractGap::WithinUnitVarianceMissing),
                    Some(v) if !(v.is_finite() && *v > 0.0) => {
                        gaps.push(ContractGap::Invalid("adaptive_inference.variance_invalid"));
                    }
                    Some(_) => {}
                }
            }
        }
        match &declaration.analysis {
            None => gaps.push(ContractGap::FinalAnalysisMissing),
            Some(analysis) => {
                let weighted_ok = match analysis.estimator {
                    AnalysisEstimator::AdaptivelyWeighted { prior, probability_floor, .. } => {
                        prior_is_valid(&prior)
                            && probability_floor.is_finite()
                            && probability_floor > 0.0
                            && probability_floor <= 1.0
                    }
                    _ => true,
                };
                if analysis.plus >= arms
                    || analysis.minus >= arms
                    || analysis.plus == analysis.minus
                    || !weighted_ok
                {
                    gaps.push(ContractGap::Invalid("adaptive_inference.analysis_invalid"));
                }
            }
        }
        match &declaration.stopping {
            None => gaps.push(ContractGap::StoppingRuleMissing),
            Some(StoppingRule::NoStopping) => {}
            Some(_) => gaps.push(ContractGap::Invalid(
                "adaptive_inference.repeated_measures_stopping_not_derived",
            )),
        }
        if !gaps.is_empty() {
            return Err(ProcedureContractError { gaps });
        }
        match (
            declaration.assignment,
            declaration.enrolment,
            declaration.structure,
            declaration.analysis,
        ) {
            (Some(assignment), Some(enrolment), Some(structure), Some(analysis)) => {
                match Covariance::build(&structure) {
                    Some(covariance) => Ok(Self {
                        arms,
                        assignment,
                        enrolment,
                        lag_looks: declaration.lag_looks,
                        structure,
                        covariance,
                        analysis,
                    }),
                    None => Err(ProcedureContractError {
                        gaps: vec![ContractGap::Invalid("adaptive_inference.correlation_invalid")],
                    }),
                }
            }
            _ => Err(ProcedureContractError { gaps }),
        }
    }

    /// Number of arms.
    #[must_use]
    pub const fn arms(&self) -> usize {
        self.arms
    }

    /// The assignment rule.
    #[must_use]
    pub const fn assignment(&self) -> &UnitAssignment {
        &self.assignment
    }

    /// Units enrolled per batch.
    #[must_use]
    pub fn enrolment(&self) -> &[u32] {
        &self.enrolment
    }

    /// The outcome structure.
    #[must_use]
    pub const fn structure(&self) -> &OutcomeStructure {
        &self.structure
    }

    /// The final analysis.
    #[must_use]
    pub const fn analysis(&self) -> &FinalAnalysis {
        &self.analysis
    }

    /// Follow-up delay in batches.
    #[must_use]
    pub const fn lag_looks(&self) -> usize {
        self.lag_looks
    }

    /// Measurements per unit (members per cluster).
    #[must_use]
    pub const fn measurements(&self) -> usize {
        self.covariance.measurements
    }

    /// Units enrolled in total.
    #[must_use]
    pub fn total_units(&self) -> u32 {
        self.enrolment.iter().sum()
    }

    /// Units whose summary is available at the final analysis: the batches
    /// enrolled at least `lag_looks` batches before the last.
    #[must_use]
    pub fn revealed_units(&self) -> u32 {
        let looks = self.enrolment.len();
        if self.lag_looks >= looks {
            0
        } else {
            self.enrolment[..looks - self.lag_looks].iter().sum()
        }
    }

    fn analysis_summary(&self) -> UnitSummary {
        match self.analysis.estimator {
            AnalysisEstimator::ClusterRobust { summary }
            | AnalysisEstimator::AdaptivelyWeighted { summary, .. } => summary,
            AnalysisEstimator::PooledIndependent => UnitSummary::Mean,
        }
    }

    /// Variance of one unit's summary under the declared structure and the
    /// analysis's unit reduction.
    #[must_use]
    pub fn unit_summary_variance(&self) -> f64 {
        self.covariance.summary_variance(self.analysis_summary())
    }

    /// Marginal variance of one measurement.
    #[must_use]
    pub fn marginal_variance(&self) -> f64 {
        self.covariance.marginal_variance
    }

    /// Variance inflation of the unit summary over `m` independent
    /// measurements: `m * Var(summary) / marginal variance`. Under compound
    /// symmetry (and for a random-effect cluster) with the mean this is
    /// `1 + (m - 1) * rho` with `rho` the intra-unit correlation.
    #[must_use]
    pub fn design_effect(&self) -> f64 {
        self.covariance.measurements as f64 * self.unit_summary_variance()
            / self.covariance.marginal_variance
    }

    /// Whether the declared dependence is zero.
    #[must_use]
    pub fn measurements_independent(&self) -> bool {
        self.covariance.is_independent()
    }

    /// The outcome-structure name of the procedure matrix.
    #[must_use]
    pub const fn outcome_name(&self) -> &'static str {
        match self.structure {
            OutcomeStructure::RepeatedMeasures { .. } => REPEATED_MEASURES,
            OutcomeStructure::Hierarchical { .. } => HIERARCHICAL,
        }
    }

    /// The assignment-family name of the procedure matrix: a batch above one
    /// unit makes an adaptive rule a batched design.
    #[must_use]
    pub fn family_name(&self) -> &'static str {
        match &self.assignment {
            UnitAssignment::Fixed { .. } => "fixed_randomization",
            UnitAssignment::EpsilonGreedy { .. } => {
                if self.enrolment.iter().any(|&n| n > 1) {
                    "batched_adaptive"
                } else {
                    "epsilon_greedy"
                }
            }
        }
    }

    /// The paradigm the declared analysis belongs to.
    #[must_use]
    pub const fn declared_paradigm(&self) -> UncertaintyParadigm {
        match self.analysis.estimator {
            AnalysisEstimator::AdaptivelyWeighted { .. } => {
                UncertaintyParadigm::FrequentistAdaptiveWeighting
            }
            _ => UncertaintyParadigm::FrequentistFixedSample,
        }
    }

    /// The analysis route label.
    #[must_use]
    pub const fn route_name(&self) -> &'static str {
        match self.analysis.estimator {
            AnalysisEstimator::ClusterRobust { .. } => "cluster_robust_unit_summaries",
            AnalysisEstimator::PooledIndependent => "pooled_independent_measurements",
            AnalysisEstimator::AdaptivelyWeighted { .. } => "adaptively_weighted_unit_scores",
        }
    }

    /// The matrix cell this procedure belongs to, with its paradigm.
    #[must_use]
    pub fn cell(&self) -> CellLabel {
        CellLabel {
            family: self.family_name(),
            outcome: self.outcome_name(),
            monitoring: "final_analysis_only",
            stopping: "no_stopping",
            paradigm: self.declared_paradigm().as_str(),
        }
    }

    fn policy_row(&self, sums: &[f64], counts: &[u32]) -> Vec<f64> {
        match &self.assignment {
            UnitAssignment::Fixed { probabilities } => probabilities.clone(),
            UnitAssignment::EpsilonGreedy { epsilon, prior } => {
                let means: Vec<f64> = (0..self.arms)
                    .map(|a| {
                        (prior.strength * prior.mean + sums[a])
                            / (prior.strength + f64::from(counts[a]))
                    })
                    .collect();
                let best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let greedy: Vec<bool> =
                    means.iter().map(|m| (best - m).abs() <= TIE_TOLERANCE).collect();
                let tied = greedy.iter().filter(|&&g| g).count() as f64;
                (0..self.arms)
                    .map(|a| {
                        epsilon / self.arms as f64
                            + if greedy[a] { (1.0 - epsilon) / tied } else { 0.0 }
                    })
                    .collect()
            }
        }
    }
}

/// The matrix cell a derived statement is about. A statement about one cell
/// is never a statement about another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellLabel {
    /// Assignment family.
    pub family: &'static str,
    /// Outcome structure.
    pub outcome: &'static str,
    /// Monitoring strategy.
    pub monitoring: &'static str,
    /// Stopping kind.
    pub stopping: &'static str,
    /// Inferential paradigm.
    pub paradigm: &'static str,
}

impl CellLabel {
    /// Stable identifier `family/outcome/monitoring/stopping`.
    #[must_use]
    pub fn id(&self) -> String {
        format!("{}/{}/{}/{}", self.family, self.outcome, self.monitoring, self.stopping)
    }

    /// Whether a guarantee derived for this cell may be stated for `other`.
    ///
    /// # Errors
    /// [`Error::PremiseFailed`] unless every axis and the paradigm are equal:
    /// a guarantee for two components never licenses their pairing, and the
    /// evidence for one cell does not transfer to another.
    pub fn transfer_to(&self, other: &Self) -> Result<(), Error> {
        self.transfer_to_named(&other.id(), other.paradigm)
    }

    /// [`Self::transfer_to`] for a target named by its identifier
    /// (`family/outcome/monitoring/stopping`) and paradigm.
    ///
    /// # Errors
    /// As [`Self::transfer_to`].
    pub fn transfer_to_named(&self, id: &str, paradigm: &str) -> Result<(), Error> {
        if self.id() == id && self.paradigm == paradigm {
            Ok(())
        } else {
            Err(Error::PremiseFailed {
                detail: "adaptive_inference.guarantee_not_transferable",
                message: "a guarantee derived for one cell of the procedure matrix is not a statement about another cell; derive it for that cell",
            })
        }
    }
}

// -- Premise gates --------------------------------------------------------------------------------

fn premise(detail: &'static str, message: &'static str) -> Error {
    Error::PremiseFailed { detail, message }
}

/// Check that the declared procedure meets the validity premises of
/// `paradigm`.
///
/// # Errors
/// [`Error::PremiseFailed`] with a stable `adaptive_inference.*` detail: a
/// fixed-sample analysis after response-adaptive assignment, an independence
/// assumption for dependent measurements, a paradigm that is not derived for
/// this cell, or an analysis whose route is not the paradigm's.
pub fn admit_repeated_measures(
    spec: &RepeatedMeasuresSpec,
    paradigm: UncertaintyParadigm,
) -> Result<(), Error> {
    if matches!(spec.analysis.estimator, AnalysisEstimator::PooledIndependent)
        && !spec.measurements_independent()
    {
        return Err(premise(
            "adaptive_inference.independence_assumed_for_dependent_measurements",
            "a pooled analysis treats the measurements of a unit as independent, which the declared within-unit dependence contradicts; analyse unit summaries with a cluster-robust variance",
        ));
    }
    let weighted = matches!(spec.analysis.estimator, AnalysisEstimator::AdaptivelyWeighted { .. });
    match paradigm {
        UncertaintyParadigm::FrequentistFixedSample => {
            if spec.assignment.is_adaptive() {
                return Err(premise(
                    "adaptive_inference.fixed_sample_design_adaptive",
                    "a fixed-sample analysis presumes assignment probabilities fixed in advance; the declared assignment adapts to observed unit summaries",
                ));
            }
            if weighted {
                return Err(premise(
                    "adaptive_inference.analysis_route_mismatch",
                    "the declared analysis is the adaptive-weighting route, not a fixed-sample analysis",
                ));
            }
        }
        UncertaintyParadigm::FrequentistAdaptiveWeighting => {
            if !weighted {
                return Err(if spec.assignment.is_adaptive() {
                    premise(
                        "adaptive_inference.fixed_sample_design_adaptive",
                        "the declared analysis is a fixed-sample contrast of unit summaries; after response-adaptive assignment only the weighting route is derived",
                    )
                } else {
                    premise(
                        "adaptive_inference.analysis_route_mismatch",
                        "the declared analysis is a fixed-sample contrast, not the adaptive-weighting route",
                    )
                });
            }
        }
        _ => {
            return Err(premise(
                "adaptive_inference.repeated_measures_paradigm_not_derived",
                "only the fixed-sample and adaptive-weighting paradigms are derived for repeated-measures and clustered outcomes; other paradigms are separate cells",
            ));
        }
    }
    Ok(())
}

// -- Exact closed form ----------------------------------------------------------------------------

/// One component of a Gaussian mixture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureComponent {
    /// Unconditional probability of the component.
    pub weight: f64,
    /// Mean.
    pub mean: f64,
    /// Variance.
    pub variance: f64,
}

/// The exact law of a fixed-allocation contrast: a mixture of normals over
/// the number of units in each arm, with the mass on which the contrast is
/// undefined (an arm without a unit).
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianMixtureLaw {
    /// Components, by number of units in the plus arm.
    pub components: Vec<MixtureComponent>,
    /// Probability that the contrast is undefined.
    pub undefined_mass: f64,
}

impl GaussianMixtureLaw {
    /// Probability that the contrast is defined.
    #[must_use]
    pub fn defined_mass(&self) -> f64 {
        self.components.iter().map(|c| c.weight).sum()
    }

    /// Mean of the contrast given that it is defined.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let mass = self.defined_mass();
        self.components.iter().map(|c| c.weight * c.mean).sum::<f64>() / mass
    }

    /// Variance of the contrast given that it is defined.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let mass = self.defined_mass();
        let mean = self.mean();
        self.components
            .iter()
            .map(|c| c.weight * (c.variance + (c.mean - mean).powi(2)))
            .sum::<f64>()
            / mass
    }

    /// Distribution function of the contrast given that it is defined.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let mass = self.defined_mass();
        self.components
            .iter()
            .map(|c| c.weight * norm_cdf((x - c.mean) / c.variance.sqrt()))
            .sum::<f64>()
            / mass
    }
}

/// The exact law of the contrast of arm means of unit summaries under a
/// fixed two-arm allocation and the arm means `arm_means`.
///
/// Given `k` units in the plus arm and `n - k` in the minus arm, the contrast
/// is normal with mean `mu_plus - mu_minus` and variance
/// `Var(unit summary) * (1 / k + 1 / (n - k))`; `k` is binomial. `n` counts
/// the units whose summary is available at the final analysis.
///
/// # Errors
/// An adaptive assignment, an analysis that is not a contrast of arm means,
/// more than two arms, or arm means of the wrong shape: those have no closed
/// form here and are simulated instead.
pub fn exact_fixed_contrast_law(
    spec: &RepeatedMeasuresSpec,
    arm_means: &[f64],
) -> Result<GaussianMixtureLaw, Error> {
    let UnitAssignment::Fixed { probabilities } = &spec.assignment else {
        return Err(premise(
            "adaptive_inference.exact_law_unavailable",
            "an adaptively assigned procedure has no closed-form contrast law; simulate the whole procedure",
        ));
    };
    if spec.arms != 2
        || matches!(spec.analysis.estimator, AnalysisEstimator::AdaptivelyWeighted { .. })
    {
        return Err(premise(
            "adaptive_inference.exact_law_unavailable",
            "the closed form covers a contrast of arm means of unit summaries over two arms",
        ));
    }
    check_means(spec, arm_means)?;
    let n = spec.revealed_units();
    let plus = spec.analysis.plus;
    let minus = spec.analysis.minus;
    let p = probabilities[plus];
    let q = probabilities[minus];
    let delta = arm_means[plus] - arm_means[minus];
    let v = spec.unit_summary_variance();
    let mut components = Vec::new();
    if n >= 2 && p > 0.0 && q > 0.0 {
        let mut pmf = (f64::from(n) * q.ln()).exp();
        for k in 0..=n {
            if k >= 1 && k < n {
                components.push(MixtureComponent {
                    weight: pmf,
                    mean: delta,
                    variance: v * (1.0 / f64::from(k) + 1.0 / f64::from(n - k)),
                });
            }
            if k < n {
                pmf *= f64::from(n - k) / f64::from(k + 1) * (p / q);
            }
        }
    }
    let defined: f64 = components.iter().map(|c| c.weight).sum();
    Ok(GaussianMixtureLaw { components, undefined_mass: 1.0 - defined })
}

fn check_means(spec: &RepeatedMeasuresSpec, arm_means: &[f64]) -> Result<(), Error> {
    if arm_means.len() != spec.arms {
        return Err(Error::Shape("one mean per arm"));
    }
    if arm_means.iter().any(|m| !m.is_finite()) {
        return Err(Error::InvalidParameter("an arm mean must be finite"));
    }
    Ok(())
}

// -- Simulation -----------------------------------------------------------------------------------

/// What to simulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatedMeasuresRequest {
    /// Replications to run.
    pub replications: u32,
    /// Seed of the replicate streams.
    pub seed: u64,
}

/// One simulated run of the whole procedure.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresReplicate {
    /// The contrast estimate; `None` unless both contrast arms have a unit.
    pub estimate: Option<f64>,
    /// Plug-in variance of the estimate (uncalibrated); `None` without two
    /// units (two measurements for the pooled analysis) per contrast arm, or
    /// without two analysed units for the weighted analysis.
    pub variance: Option<f64>,
    /// Units enrolled.
    pub enrolled: u32,
    /// Units whose summary entered the analysis.
    pub analysed_units: usize,
    /// Analysed units per arm.
    pub units_per_arm: Vec<u32>,
}

/// The collection of replicates.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresLaw {
    /// Replicates, in index order.
    pub replicates: Vec<RepeatedMeasuresReplicate>,
}

/// Summary of the simulated estimates against the known contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepeatedMeasuresSummary {
    /// Replicates with a defined estimate.
    pub defined: usize,
    /// Replicates with a defined estimate and variance.
    pub with_variance: usize,
    /// Fraction of replicates whose estimate is undefined.
    pub undefined_fraction: f64,
    /// Mean estimate.
    pub mean: f64,
    /// Sample variance of the estimates.
    pub variance: f64,
    /// Mean estimate minus the truth.
    pub bias: f64,
    /// Monte Carlo standard error of the mean estimate.
    pub monte_carlo_standard_error: f64,
    /// Mean plug-in variance over the replicates that have one.
    pub mean_reported_variance: f64,
    /// Fraction of replicates with a variance whose plug-in Wald interval
    /// contains the truth: a measured property of this simulation, not a
    /// coverage statement.
    pub wald_contains_truth: f64,
}

impl RepeatedMeasuresLaw {
    /// Summarize against `truth` with the Wald multiplier `z`.
    ///
    /// # Errors
    /// Fewer than two defined estimates, or a non-finite truth, or a
    /// non-positive multiplier.
    pub fn summarize(&self, truth: f64, z: f64) -> Result<RepeatedMeasuresSummary, Error> {
        if !(truth.is_finite() && z.is_finite() && z > 0.0) {
            return Err(Error::InvalidParameter("truth and multiplier must be finite, z positive"));
        }
        let estimates: Vec<f64> = self.replicates.iter().filter_map(|r| r.estimate).collect();
        if estimates.len() < 2 {
            return Err(Error::Shape("a summary needs at least two defined replicates"));
        }
        let with_variance: Vec<(f64, f64)> =
            self.replicates.iter().filter_map(|r| r.estimate.zip(r.variance)).collect();
        let n = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / n;
        let variance = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let reported = if with_variance.is_empty() {
            f64::NAN
        } else {
            with_variance.iter().map(|(_, v)| v).sum::<f64>() / with_variance.len() as f64
        };
        let contained =
            with_variance.iter().filter(|(e, v)| (e - truth).abs() <= z * v.sqrt()).count();
        Ok(RepeatedMeasuresSummary {
            defined: estimates.len(),
            with_variance: with_variance.len(),
            undefined_fraction: 1.0 - n / self.replicates.len() as f64,
            mean,
            variance,
            bias: mean - truth,
            monte_carlo_standard_error: (variance / n).sqrt(),
            mean_reported_variance: reported,
            wald_contains_truth: if with_variance.is_empty() {
                f64::NAN
            } else {
                contained as f64 / with_variance.len() as f64
            },
        })
    }
}

fn draw_arm(rng: &mut CausalRng, probabilities: &[f64]) -> usize {
    let u = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, &p) in probabilities.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

struct Unit {
    arm: usize,
    row: Vec<f64>,
    summary: f64,
    values: Vec<f64>,
}

fn mean_and_variance(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    Some((mean, values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)))
}

fn analyse(
    spec: &RepeatedMeasuresSpec,
    units: &[Unit],
    revealed: &[usize],
    cancelled: &dyn Fn() -> bool,
) -> Result<(Option<f64>, Option<f64>), Error> {
    let plus = spec.analysis.plus;
    let minus = spec.analysis.minus;
    match spec.analysis.estimator {
        AnalysisEstimator::ClusterRobust { .. } => {
            let group = |arm: usize| -> Vec<f64> {
                revealed
                    .iter()
                    .filter(|&&i| units[i].arm == arm)
                    .map(|&i| units[i].summary)
                    .collect()
            };
            let (a, b) = (group(plus), group(minus));
            if a.is_empty() || b.is_empty() {
                return Ok((None, None));
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let estimate = mean(&a) - mean(&b);
            let variance = mean_and_variance(&a)
                .zip(mean_and_variance(&b))
                .map(|((_, va), (_, vb))| va / a.len() as f64 + vb / b.len() as f64);
            Ok((Some(estimate), variance))
        }
        AnalysisEstimator::PooledIndependent => {
            let pool = |arm: usize| -> Vec<f64> {
                revealed
                    .iter()
                    .filter(|&&i| units[i].arm == arm)
                    .flat_map(|&i| units[i].values.iter().copied())
                    .collect()
            };
            let (a, b) = (pool(plus), pool(minus));
            if a.is_empty() || b.is_empty() {
                return Ok((None, None));
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let estimate = mean(&a) - mean(&b);
            let variance = mean_and_variance(&a)
                .zip(mean_and_variance(&b))
                .map(|((_, va), (_, vb))| va / a.len() as f64 + vb / b.len() as f64);
            Ok((Some(estimate), variance))
        }
        AnalysisEstimator::AdaptivelyWeighted { scheme, prior, probability_floor, .. } => {
            if revealed.len() < 2 {
                return Ok((None, None));
            }
            let assigned: Vec<usize> = revealed.iter().map(|&i| units[i].arm).collect();
            let probabilities: Vec<f64> =
                revealed.iter().flat_map(|&i| units[i].row.iter().copied()).collect();
            let outcomes: Vec<f64> = revealed.iter().map(|&i| units[i].summary).collect();
            let scores =
                adaptive_scores(spec.arms, &assigned, &probabilities, &outcomes, prior, cancelled)?;
            let fit = weighted_estimate(
                spec.arms,
                &probabilities,
                &scores,
                Estimand::Contrast { plus, minus },
                scheme,
                None,
                probability_floor,
            )?;
            Ok((Some(fit.estimate), Some(fit.variance)))
        }
    }
}

/// Run the declared procedure `request.replications` times under the arm
/// means `arm_means`.
///
/// Replicate streams are keyed by the seed and the replicate index, so a
/// replicate does not depend on how many others are run. The policy sees only
/// the unit summaries of units whose follow-up has completed; the analysis
/// uses the same units.
///
/// # Errors
/// A premise failure of the declared paradigm, arm means of the wrong shape,
/// a request outside bounds, a positivity failure of the weighted analysis,
/// or cancellation.
pub fn simulate_repeated_measures(
    spec: &RepeatedMeasuresSpec,
    arm_means: &[f64],
    request: &RepeatedMeasuresRequest,
    cancelled: &dyn Fn() -> bool,
) -> Result<RepeatedMeasuresLaw, Error> {
    admit_repeated_measures(spec, spec.declared_paradigm())?;
    check_means(spec, arm_means)?;
    if request.replications == 0 || request.replications > MAX_SIMULATION_REPLICATIONS {
        return Err(Error::Limit("between 1 and 1,000,000 replications"));
    }
    let summary_kind = spec.analysis_summary();
    let mut replicates = Vec::with_capacity(request.replications as usize);
    for replicate in 0..request.replications {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut rng = CausalRng::from_seed(
            request.seed ^ (u64::from(replicate) + 1).wrapping_mul(STREAM_MIX),
        );
        let mut sums = vec![0.0; spec.arms];
        let mut counts = vec![0_u32; spec.arms];
        let mut units: Vec<Unit> = Vec::with_capacity(spec.total_units() as usize);
        let mut pending: VecDeque<Vec<usize>> = VecDeque::new();
        let mut revealed: Vec<usize> = Vec::new();
        for &size in &spec.enrolment {
            let row = spec.policy_row(&sums, &counts);
            let mut batch = Vec::with_capacity(size as usize);
            for _ in 0..size {
                let arm = draw_arm(&mut rng, &row);
                let values = spec.covariance.draw(&mut rng, arm_means[arm]);
                let summary = spec.covariance.reduce(&values, summary_kind);
                batch.push(units.len());
                units.push(Unit { arm, row: row.clone(), summary, values });
            }
            pending.push_back(batch);
            if pending.len() > spec.lag_looks {
                if let Some(aged) = pending.pop_front() {
                    for index in aged {
                        sums[units[index].arm] += units[index].summary;
                        counts[units[index].arm] += 1;
                        revealed.push(index);
                    }
                }
            }
        }
        let (estimate, variance) = analyse(spec, &units, &revealed, cancelled)?;
        replicates.push(RepeatedMeasuresReplicate {
            estimate,
            variance,
            enrolled: spec.total_units(),
            analysed_units: revealed.len(),
            units_per_arm: counts,
        });
    }
    Ok(RepeatedMeasuresLaw { replicates })
}

// -- Derivation -----------------------------------------------------------------------------------

/// The derived uncertainty of a repeated-measures or clustered procedure:
/// exact closed forms and the simulated law with its Monte Carlo error,
/// reported separately.
#[derive(Clone, Debug, PartialEq)]
pub struct RepeatedMeasuresDerivation {
    /// The matrix cell the statement is about.
    pub cell: CellLabel,
    /// The paradigm.
    pub paradigm: UncertaintyParadigm,
    /// The analysis route label.
    pub route: &'static str,
    /// The contrast under the arm means.
    pub contrast_value: f64,
    /// Exact variance of one unit's summary.
    pub unit_summary_variance: f64,
    /// Exact design effect of the unit summary.
    pub design_effect: f64,
    /// Exact contrast law (fixed two-arm allocation, arm-mean contrast);
    /// `None` when the law has no closed form.
    pub exact_law: Option<GaussianMixtureLaw>,
    /// Simulated estimates against the contrast, with Monte Carlo error.
    pub simulated: RepeatedMeasuresSummary,
    /// Replications run.
    pub replications: u32,
    /// Seed of the replicate streams.
    pub seed: u64,
    /// Mean enrolled units.
    pub enrolled: u32,
    /// Mean analysed units per replicate.
    pub mean_analysed_units: f64,
    /// Always [`DERIVED_UNMEASURED`].
    pub calibration: &'static str,
}

/// Derive the uncertainty of the declared procedure under `arm_means`.
///
/// # Errors
/// A premise failure, arm means of the wrong shape, a request outside
/// bounds, a positivity failure, fewer than two defined replicates, or
/// cancellation.
pub fn derive_repeated_measures(
    spec: &RepeatedMeasuresSpec,
    arm_means: &[f64],
    request: &RepeatedMeasuresRequest,
    z: f64,
    cancelled: &dyn Fn() -> bool,
) -> Result<RepeatedMeasuresDerivation, Error> {
    let paradigm = spec.declared_paradigm();
    admit_repeated_measures(spec, paradigm)?;
    let law = simulate_repeated_measures(spec, arm_means, request, cancelled)?;
    let contrast_value = arm_means[spec.analysis.plus] - arm_means[spec.analysis.minus];
    let simulated = law.summarize(contrast_value, z)?;
    let exact_law = exact_fixed_contrast_law(spec, arm_means).ok();
    let n = law.replicates.len() as f64;
    Ok(RepeatedMeasuresDerivation {
        cell: spec.cell(),
        paradigm,
        route: spec.route_name(),
        contrast_value,
        unit_summary_variance: spec.unit_summary_variance(),
        design_effect: spec.design_effect(),
        exact_law,
        simulated,
        replications: request.replications,
        seed: request.seed,
        enrolled: spec.total_units(),
        mean_analysed_units: law.replicates.iter().map(|r| r.analysed_units as f64).sum::<f64>()
            / n,
        calibration: DERIVED_UNMEASURED,
    })
}
