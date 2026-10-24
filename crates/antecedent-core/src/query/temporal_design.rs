//! Sequentially randomized unit-period designs and regime survival on subject trajectories.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::QueryError;
use crate::canonical_encode::{
    CanonicalEncode, count, encode_flags, encode_i64s, encode_usizes, encode_variables, flag,
};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Target of a continuous-dose design.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalDoseTarget {
    /// Kernel-smoothed dose response at every grid dose; the bandwidth is part of the target.
    SmoothedCurve {
        /// Grid doses.
        grid: Arc<[f64]>,
    },
    /// Contrast of the kernel-smoothed response between two doses.
    SmoothedContrast {
        /// Active dose.
        active: f64,
        /// Control dose.
        control: f64,
    },
    /// Local-quadratic level (or first derivative) of the dose response on a grid; the
    /// interval accounts for the smoothing bias under the declared third-derivative bound.
    LocalQuadratic {
        /// Grid doses.
        grid: Arc<[f64]>,
        /// Report the first derivative instead of the level.
        derivative: bool,
        /// Declared bound on `|m'''|`; required for a frequentist interval.
        curvature_bound: Option<f64>,
    },
}

/// How the value of a whole static treatment regime is estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalRegimeMethod {
    /// Iterated-conditional-expectation g-formula.
    GFormula,
    /// Inverse-probability-weighted saturated marginal structural model of each path.
    InverseProbability,
    /// Recursive augmented inverse-probability estimator, consistent when either the outcome
    /// regressions or the treatment models are correct.
    DoublyRobust,
}

impl TemporalRegimeMethod {
    /// The stable name used by the wire and the Python surface.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::GFormula => "g_formula",
            Self::InverseProbability => "inverse_probability_weighting",
            Self::DoublyRobust => "doubly_robust",
        }
    }

    /// The method named `name`, if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::GFormula, Self::InverseProbability, Self::DoublyRobust]
            .into_iter()
            .find(|method| method.name() == name)
    }
}

/// The assignment record of a sequentially randomized design, one entry per table row.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalAssignment {
    /// Two arms: the realized assignment and its known probability of treatment.
    TwoArm {
        /// Realized assignment in row order.
        assignment: Arc<[bool]>,
        /// Known `P(A_t = 1 | history)` in row order.
        probability: Arc<[f64]>,
    },
    /// Three or more arms or categorical levels; the first is the reference.
    MultiArm {
        /// Realized arm index in row order.
        assignment: Arc<[usize]>,
        /// Known probability of every arm, one row per table row.
        probabilities: Arc<[Vec<f64>]>,
        /// Arm labels; the first is the reference.
        arms: Arc<[Arc<str>]>,
        /// Categorical levels: the contrasts form one Holm family.
        categorical: bool,
        /// Ordered levels: adjacent steps are reported beside the contrasts.
        ordered: bool,
    },
    /// Two binary factors assigned independently at each decision.
    Factorial {
        /// First factor in row order.
        first: Arc<[bool]>,
        /// Known probability of the first factor.
        first_probability: Arc<[f64]>,
        /// Second factor in row order.
        second: Arc<[bool]>,
        /// Known probability of the second factor.
        second_probability: Arc<[f64]>,
    },
    /// Two arms with spillover inside declared interference clusters.
    Interference {
        /// Realized assignment in row order.
        assignment: Arc<[bool]>,
        /// Known probability of treatment in row order.
        probability: Arc<[f64]>,
        /// Interference cluster of each row.
        clusters: Arc<[Arc<str>]>,
    },
    /// A continuous dose with the known density of each realized dose.
    Dose {
        /// Realized dose in row order.
        dose: Arc<[f64]>,
        /// Density of the realized dose given the history, in row order.
        density: Arc<[f64]>,
        /// Gaussian kernel bandwidth.
        bandwidth: f64,
        /// Dose target.
        target: TemporalDoseTarget,
    },
    /// Static treatment regimes on observational subject trajectories with time-varying
    /// confounding; the query outcome is the event indicator column.
    RegimeSurvival {
        /// Treatment indicator column.
        treatment: VariableId,
        /// Censoring indicator column.
        censored: VariableId,
        /// Time-varying confounder columns measured at the start of each period.
        confounders: Arc<[VariableId]>,
        /// Survival horizon in periods.
        horizon: u32,
    },
    /// Observational decisions: a binary treatment whose assignment law is not recorded and
    /// is estimated from the measured decision-time history. The excursion contrast is
    /// identified by declared sequential exchangeability given that history, not by design.
    ObservationalExcursion {
        /// Binary treatment indicator column, coded exactly 0 or 1.
        treatment: VariableId,
        /// History columns measured at each decision that the assignment model reads.
        history: Arc<[VariableId]>,
    },
    /// The value of whole static treatment regimes on a continuous end-of-follow-up outcome
    /// under time-varying confounding, identified by sequential exchangeability given the
    /// measured confounder and treatment history. Every unit is observed in the same periods;
    /// the outcome column is read at each unit's last period, and the contrast is the mean
    /// outcome under the active treatment path minus that under the reference path.
    ObservationalRegime {
        /// Binary treatment indicator column, coded exactly 0 or 1.
        treatment: VariableId,
        /// Time-varying confounder columns measured at the start of each period.
        confounders: Arc<[VariableId]>,
        /// The active path: the treatment taken in each period, in period order.
        active: Arc<[bool]>,
        /// The reference path, in period order.
        reference: Arc<[bool]>,
        /// How the value of each path is estimated.
        method: TemporalRegimeMethod,
    },
}

/// Validation-sample counts of a misclassified binary outcome whose error is the same in every
/// arm, unit and period (non-differential).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutcomeMisclassification {
    /// Recorded positive among true positives.
    pub true_positive: u64,
    /// Recorded negative among true positives.
    pub false_negative: u64,
    /// Recorded negative among true negatives.
    pub true_negative: u64,
    /// Recorded positive among true negatives.
    pub false_positive: u64,
}

/// The outcome functional of an arm design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalDesignFunctional {
    /// Mean outcome per arm and their contrasts.
    Mean,
    /// The `level`-quantile of the outcome law per arm and its difference (two arms only).
    Quantile {
        /// Quantile level in `(0, 1)`.
        level: f64,
    },
}

/// Excursion effects of a sequentially randomized unit-period design, or regime survival on
/// subject trajectories.
///
/// One table row per unit and period. Unit and period labels and the assignment record are part
/// of the query; the outcome (and, for regime survival, the treatment, censoring and confounder
/// columns) are table columns. The target pairs each decision with the same unit's outcome
/// `lag` periods later. Identification is by the recorded sequential randomization (regime
/// survival: declared sequential exchangeability given the confounders); none of it is derived.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalDesignQuery {
    /// Outcome column (the event indicator for regime survival).
    pub outcome: VariableId,
    /// Unit label of each row.
    pub units: Arc<[Arc<str>]>,
    /// Period of each row; `(unit, period)` is unique.
    pub periods: Arc<[i64]>,
    /// Periods from a decision to its outcome.
    pub lag: u32,
    /// Assignment record.
    pub assignment: TemporalAssignment,
    /// Outcome functional.
    pub functional: TemporalDesignFunctional,
    /// Positivity floor of every realized assignment probability, in `(0, 0.5]`.
    pub min_probability: f64,
    /// Validation counts of a misclassified binary outcome; `None` when the outcome is
    /// recorded without error.
    pub outcome_misclassification: Option<OutcomeMisclassification>,
}

impl CanonicalEncode for TemporalDoseTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::SmoothedCurve { grid } => {
                writer.word(1).reals(grid);
            }
            Self::SmoothedContrast { active, control } => {
                writer.word(2).real(*active).real(*control);
            }
            Self::LocalQuadratic { grid, derivative, curvature_bound } => {
                writer.word(3).reals(grid).word(flag(*derivative)).opt_real(*curvature_bound);
            }
        }
    }
}

/// Every array is row-aligned and keeps its order; arm labels keep their positions.
impl CanonicalEncode for TemporalAssignment {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::TwoArm { assignment, probability } => {
                writer.word(1);
                encode_flags(writer, assignment);
                writer.reals(probability);
            }
            Self::MultiArm { assignment, probabilities, arms, categorical, ordered } => {
                writer.word(2);
                encode_usizes(writer, assignment);
                writer.word(count(probabilities.len()));
                for row in probabilities.iter() {
                    writer.reals(row);
                }
                writer.texts(arms).word(flag(*categorical)).word(flag(*ordered));
            }
            Self::Factorial { first, first_probability, second, second_probability } => {
                writer.word(3);
                encode_flags(writer, first);
                writer.reals(first_probability);
                encode_flags(writer, second);
                writer.reals(second_probability);
            }
            Self::Interference { assignment, probability, clusters } => {
                writer.word(4);
                encode_flags(writer, assignment);
                writer.reals(probability).texts(clusters);
            }
            Self::Dose { dose, density, bandwidth, target } => {
                writer.word(5).reals(dose).reals(density).real(*bandwidth);
                target.encode(writer);
            }
            Self::RegimeSurvival { treatment, censored, confounders, horizon } => {
                writer.word(6);
                treatment.encode(writer);
                censored.encode(writer);
                encode_variables(writer, confounders);
                writer.word(u64::from(*horizon));
            }
            Self::ObservationalExcursion { treatment, history } => {
                writer.word(7);
                treatment.encode(writer);
                encode_variables(writer, history);
            }
            Self::ObservationalRegime { treatment, confounders, active, reference, method } => {
                writer.word(8);
                treatment.encode(writer);
                encode_variables(writer, confounders);
                encode_flags(writer, active);
                encode_flags(writer, reference);
                writer.word(match method {
                    TemporalRegimeMethod::GFormula => 1,
                    TemporalRegimeMethod::InverseProbability => 2,
                    TemporalRegimeMethod::DoublyRobust => 3,
                });
            }
        }
    }
}

impl CanonicalEncode for TemporalDesignFunctional {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Mean => {
                writer.word(1);
            }
            Self::Quantile { level } => {
                writer.word(2).real(*level);
            }
        }
    }
}

/// Every field is written. The row-aligned labels keep their order.
impl CanonicalEncode for TemporalDesignQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        writer.texts(&self.units);
        encode_i64s(writer, &self.periods);
        writer.word(u64::from(self.lag));
        self.assignment.encode(writer);
        self.functional.encode(writer);
        writer.real(self.min_probability);
        match &self.outcome_misclassification {
            Some(counts) => {
                writer
                    .word(1)
                    .word(counts.true_positive)
                    .word(counts.false_negative)
                    .word(counts.true_negative)
                    .word(counts.false_positive);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

fn invalid(message: &str) -> QueryError {
    QueryError::InvalidTemporalDesign(message.to_owned())
}

fn probabilities_valid(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0)
}

fn grid_valid(grid: &[f64]) -> bool {
    !grid.is_empty() && grid.iter().all(|value| value.is_finite())
}

impl TemporalDesignQuery {
    /// Number of table rows the query's arrays describe.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.units.len()
    }

    /// The table columns the query reads.
    #[must_use]
    pub fn columns(&self) -> Vec<VariableId> {
        let mut columns = vec![self.outcome];
        if let TemporalAssignment::RegimeSurvival { treatment, censored, confounders, .. } =
            &self.assignment
        {
            columns.push(*treatment);
            columns.push(*censored);
            columns.extend(confounders.iter().copied());
        }
        if let TemporalAssignment::ObservationalExcursion { treatment, history } = &self.assignment
        {
            columns.push(*treatment);
            columns.extend(history.iter().copied());
        }
        if let TemporalAssignment::ObservationalRegime { treatment, confounders, .. } =
            &self.assignment
        {
            columns.push(*treatment);
            columns.extend(confounders.iter().copied());
        }
        columns
    }

    /// A short design name: `two_arm`, `multi_arm`, `categorical`, `factorial`,
    /// `interference`, `dose`, `regime_survival`, `observational_excursion` or
    /// `observational_regime`.
    #[must_use]
    pub const fn design_name(&self) -> &'static str {
        match &self.assignment {
            TemporalAssignment::TwoArm { .. } => "two_arm",
            TemporalAssignment::MultiArm { categorical: true, .. } => "categorical",
            TemporalAssignment::MultiArm { .. } => "multi_arm",
            TemporalAssignment::Factorial { .. } => "factorial",
            TemporalAssignment::Interference { .. } => "interference",
            TemporalAssignment::Dose { .. } => "dose",
            TemporalAssignment::RegimeSurvival { .. } => "regime_survival",
            TemporalAssignment::ObservationalExcursion { .. } => "observational_excursion",
            TemporalAssignment::ObservationalRegime { .. } => "observational_regime",
        }
    }

    /// Validate alignment, probabilities, labels and the functional before compiling a study.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidTemporalDesign`] for misaligned arrays, an empty or
    /// repeated `(unit, period)` label, a probability outside `(0, 1)`, a floor outside
    /// `(0, 0.5]`, a dose target with no grid or a non-positive bandwidth, a quantile functional
    /// outside a two-arm design or with a level outside `(0, 1)`, or a survival design with a
    /// lag, a zero horizon, repeated columns or a quantile functional.
    #[allow(clippy::too_many_lines, reason = "one validation arm per assignment design")]
    #[allow(clippy::float_cmp, reason = "two declared contrast doses must differ exactly")]
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.units.len();
        if n < 2 || self.periods.len() != n {
            return Err(invalid("unit and period labels must align with at least two rows"));
        }
        if self.units.iter().any(|unit| unit.trim().is_empty()) {
            return Err(invalid("unit labels must be non-empty"));
        }
        let mut seen = std::collections::HashSet::with_capacity(n);
        for (unit, period) in self.units.iter().zip(self.periods.iter()) {
            if !seen.insert((&**unit, *period)) {
                return Err(invalid("each (unit, period) must appear on exactly one row"));
            }
        }
        if !(self.min_probability.is_finite()
            && self.min_probability > 0.0
            && self.min_probability <= 0.5)
        {
            return Err(invalid("min_probability must be in (0, 0.5]"));
        }
        if let Some(counts) = &self.outcome_misclassification {
            let arm_design = matches!(
                self.assignment,
                TemporalAssignment::TwoArm { .. }
                    | TemporalAssignment::MultiArm { .. }
                    | TemporalAssignment::Factorial { .. }
                    | TemporalAssignment::Interference { .. }
            );
            if !arm_design || self.functional != TemporalDesignFunctional::Mean {
                return Err(invalid(
                    "an outcome misclassification correction applies to the mean of an arm design",
                ));
            }
            if counts.true_positive + counts.false_negative == 0
                || counts.true_negative + counts.false_positive == 0
            {
                return Err(invalid(
                    "the validation counts need true positives and true negatives",
                ));
            }
        }
        if let TemporalDesignFunctional::Quantile { level } = self.functional {
            if !(level.is_finite() && level > 0.0 && level < 1.0) {
                return Err(invalid("the quantile level must be in (0, 1)"));
            }
            if !matches!(self.assignment, TemporalAssignment::TwoArm { .. }) {
                return Err(invalid("the quantile functional is defined for a two-arm design"));
            }
        }
        match &self.assignment {
            TemporalAssignment::TwoArm { assignment, probability } => {
                if assignment.len() != n
                    || probability.len() != n
                    || !probabilities_valid(probability)
                {
                    return Err(invalid(
                        "two-arm assignment and probabilities must align with the rows and lie in (0, 1)",
                    ));
                }
            }
            TemporalAssignment::MultiArm {
                assignment,
                probabilities,
                arms,
                ordered,
                categorical,
            } => {
                if arms.len() < 3
                    || arms.iter().any(|arm| arm.trim().is_empty())
                    || arms.iter().collect::<std::collections::HashSet<_>>().len() != arms.len()
                {
                    return Err(invalid("declare at least three distinct non-empty arm labels"));
                }
                if *ordered && !*categorical {
                    return Err(invalid("ordered levels are a categorical design"));
                }
                if assignment.len() != n
                    || probabilities.len() != n
                    || assignment.iter().any(|arm| *arm >= arms.len())
                    || probabilities.iter().any(|row| {
                        row.len() != arms.len()
                            || !probabilities_valid(row)
                            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-8
                    })
                {
                    return Err(invalid(
                        "every row needs a realized arm in range and a probability row over every arm summing to one",
                    ));
                }
            }
            TemporalAssignment::Factorial {
                first,
                first_probability,
                second,
                second_probability,
            } => {
                if first.len() != n
                    || second.len() != n
                    || first_probability.len() != n
                    || second_probability.len() != n
                    || !probabilities_valid(first_probability)
                    || !probabilities_valid(second_probability)
                {
                    return Err(invalid(
                        "both factors and their probabilities must align with the rows and lie in (0, 1)",
                    ));
                }
            }
            TemporalAssignment::Interference { assignment, probability, clusters } => {
                if assignment.len() != n
                    || probability.len() != n
                    || clusters.len() != n
                    || !probabilities_valid(probability)
                    || clusters.iter().any(|cluster| cluster.trim().is_empty())
                {
                    return Err(invalid(
                        "assignment, probabilities and non-empty cluster labels must align with the rows",
                    ));
                }
            }
            TemporalAssignment::Dose { dose, density, bandwidth, target } => {
                if dose.len() != n
                    || density.len() != n
                    || dose.iter().any(|value| !value.is_finite())
                    || density.iter().any(|value| !(value.is_finite() && *value > 0.0))
                    || !(bandwidth.is_finite() && *bandwidth > 0.0)
                {
                    return Err(invalid(
                        "doses and positive densities must align with the rows, and the bandwidth must be positive",
                    ));
                }
                let target_valid = match target {
                    TemporalDoseTarget::SmoothedCurve { grid } => grid_valid(grid),
                    TemporalDoseTarget::SmoothedContrast { active, control } => {
                        active.is_finite() && control.is_finite() && active != control
                    }
                    TemporalDoseTarget::LocalQuadratic { grid, curvature_bound, .. } => {
                        grid_valid(grid)
                            && curvature_bound.is_none_or(|bound| bound.is_finite() && bound >= 0.0)
                    }
                };
                if !target_valid {
                    return Err(invalid(
                        "a dose target needs a finite grid, two distinct doses, or a non-negative curvature bound",
                    ));
                }
            }
            TemporalAssignment::RegimeSurvival { treatment, censored, confounders, horizon } => {
                if self.lag != 0 || *horizon == 0 {
                    return Err(invalid(
                        "regime survival reads each period's event (lag 0) up to a positive horizon",
                    ));
                }
                let mut columns = vec![self.outcome, *treatment, *censored];
                columns.extend(confounders.iter().copied());
                let mut sorted = columns.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != columns.len() {
                    return Err(invalid(
                        "event, treatment, censoring and confounder columns must be distinct",
                    ));
                }
            }
            TemporalAssignment::ObservationalExcursion { treatment, history } => {
                if self.outcome_misclassification.is_some() {
                    return Err(invalid(
                        "an outcome misclassification correction applies to the recorded-law arm designs",
                    ));
                }
                let mut columns = vec![self.outcome, *treatment];
                columns.extend(history.iter().copied());
                let mut sorted = columns.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != columns.len() {
                    return Err(invalid(
                        "outcome, treatment and history columns of an observational excursion must be distinct",
                    ));
                }
            }
            TemporalAssignment::ObservationalRegime {
                treatment,
                confounders,
                active,
                reference,
                ..
            } => {
                if self.outcome_misclassification.is_some()
                    || self.functional != TemporalDesignFunctional::Mean
                {
                    return Err(invalid(
                        "a whole-regime contrast reads the mean of a continuous outcome without a \
                         misclassification correction",
                    ));
                }
                if self.lag != 0 {
                    return Err(invalid(
                        "a whole-regime contrast reads the outcome at the end of follow-up (lag 0)",
                    ));
                }
                if active.is_empty() || active.len() != reference.len() || active == reference {
                    return Err(invalid(
                        "the active and reference regimes name one action per period, the same \
                         number of periods each, and differ in some period",
                    ));
                }
                let mut columns = vec![self.outcome, *treatment];
                columns.extend(confounders.iter().copied());
                let mut sorted = columns.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != columns.len() {
                    return Err(invalid(
                        "outcome, treatment and confounder columns of a whole-regime contrast must be distinct",
                    ));
                }
            }
        }
        Ok(())
    }
}
