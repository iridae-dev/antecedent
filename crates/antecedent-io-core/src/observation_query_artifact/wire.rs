//! Wire declarations and replay comparison for observation queries.

use antecedent_model::observation_uncertainty::{RedrawPrior, RedrawTarget};
use serde::{Deserialize, Serialize};

use crate::measurement_model_wire::MeasurementModelWire;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};

// ----------------------------------------------------------------- the wire

/// A recorded value of an indicator, with the measurement model that relates
/// the indicator to the construct it is offered for.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredSlotWire {
    /// The model variable that is the construct.
    pub construct: String,
    /// Stable identity of the recorded column.
    pub indicator_id: String,
    /// Kind of the recorded column (a `MeasurementKind` name).
    pub indicator_kind: String,
    /// The recorded level, in the construct's level coding.
    pub recorded_level: String,
    /// For a binary indicator, the positive level of the construct.
    pub positive_level: Option<String>,
    /// Coordinate system of the recorded column.
    pub representation_system: String,
    /// Version of that coordinate system.
    pub representation_version: u32,
    /// The measurement model.
    pub model: MeasurementModelWire,
}

/// The data one mechanism's numbers were fitted from, retained for redraw.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RetainedMechanismWire {
    /// Sufficient statistics of a regression on the mechanism's declared
    /// parents, intercept first.
    Gaussian {
        /// The variable the mechanism produces.
        variable: String,
        /// The regressors after the intercept, in the order of the statistics.
        parents: Vec<String>,
        /// Row-major Gram matrix `X'X`, as bit patterns.
        gram_bits: Vec<u64>,
        /// `X'y`, as bit patterns.
        xty_bits: Vec<u64>,
        /// `y'y`, as bits.
        yty_bits: u64,
        /// Rows the statistics were computed from.
        rows: u64,
    },
    /// Counts of each parent configuration of a finite mechanism.
    Table {
        /// The variable the mechanism produces.
        variable: String,
        /// `(parent configuration, (level, count))`, one entry per configuration.
        #[allow(
            clippy::type_complexity,
            reason = "the public versioned wire keeps its parent-configuration and level/count tuple shape; a one-use alias would hide this artifact contract"
        )]
        rows: Vec<(Vec<(String, String)>, Vec<(String, u64)>)>,
    },
}

impl RetainedMechanismWire {
    fn variable(&self) -> &str {
        match self {
            Self::Gaussian { variable, .. } | Self::Table { variable, .. } => variable,
        }
    }
}

/// The priors of a redraw, as bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedrawPriorWire {
    /// Dirichlet concentration added to every cell count.
    pub dirichlet_alpha_bits: u64,
    /// Prior precision of every regression coefficient relative to the error variance.
    pub coefficient_precision_bits: u64,
    /// Inverse-Gamma shape of the error variance.
    pub variance_shape_bits: u64,
    /// Inverse-Gamma scale of the error variance.
    pub variance_scale_bits: u64,
}

impl From<&RedrawPrior> for RedrawPriorWire {
    fn from(prior: &RedrawPrior) -> Self {
        Self {
            dirichlet_alpha_bits: prior.dirichlet_alpha.to_bits(),
            coefficient_precision_bits: prior.coefficient_precision.to_bits(),
            variance_shape_bits: prior.variance_shape.to_bits(),
            variance_scale_bits: prior.variance_scale.to_bits(),
        }
    }
}

impl RedrawPriorWire {
    pub(super) fn to_prior(&self) -> RedrawPrior {
        RedrawPrior {
            dirichlet_alpha: f64::from_bits(self.dirichlet_alpha_bits),
            coefficient_precision: f64::from_bits(self.coefficient_precision_bits),
            variance_shape: f64::from_bits(self.variance_shape_bits),
            variance_scale: f64::from_bits(self.variance_scale_bits),
        }
    }
}

/// The conditional functional whose parameter uncertainty is redrawn.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RedrawTargetWire {
    /// Conditional mean and variance of a continuous outcome.
    Gaussian {
        /// Outcome variable.
        outcome: String,
        /// `(variable, value bits)` set by intervention.
        interventions: Vec<(String, u64)>,
        /// `(variable, value bits)` recorded.
        observed: Vec<(String, u64)>,
    },
    /// The conditional probability of every level of a finite outcome.
    Finite {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
        /// `(variable, level)` recorded.
        observed: Vec<(String, String)>,
    },
}

impl RedrawTargetWire {
    pub(super) fn to_target(&self) -> RedrawTarget {
        match self {
            Self::Gaussian { outcome, interventions, observed } => RedrawTarget::Gaussian {
                outcome: outcome.clone(),
                interventions: real_pairs(interventions),
                observed: real_pairs(observed),
            },
            Self::Finite { outcome, interventions, observed } => RedrawTarget::Finite {
                outcome: outcome.clone(),
                interventions: interventions.clone(),
                observed: observed.clone(),
            },
        }
    }

    pub(super) fn outcome(&self) -> &str {
        match self {
            Self::Gaussian { outcome, .. } | Self::Finite { outcome, .. } => outcome,
        }
    }

    pub(super) fn has_interventions(&self) -> bool {
        match self {
            Self::Gaussian { interventions, .. } => !interventions.is_empty(),
            Self::Finite { interventions, .. } => !interventions.is_empty(),
        }
    }
}

fn real_pairs(pairs: &[(String, u64)]) -> Vec<(String, f64)> {
    pairs.iter().map(|(n, b)| (n.clone(), f64::from_bits(*b))).collect()
}

pub(super) fn real_refs(pairs: &[(String, u64)]) -> Vec<(&str, f64)> {
    pairs.iter().map(|(n, b)| (n.as_str(), f64::from_bits(*b))).collect()
}

/// What is asked.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "wire shape: boxing would change the serialized form and the public variant shape"
)]
pub enum ObservationQueryWire {
    /// `P(outcome | observed values)` on a composed finite model, optionally
    /// with one measured signal inverted through its measurement model.
    ConditionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` recorded without error.
        observed: Vec<(String, String)>,
        /// One recorded indicator value, applied through its model.
        measured: Option<MeasuredSlotWire>,
    },
    /// `P(outcome | do(interventions))` on a composed finite model.
    InterventionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
    },
    /// Both laws of one assignment from one model, reported side by side.
    ConditioningVersusIntervention {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)`, read once as observed and once as set.
        assignment: Vec<(String, String)>,
    },
    /// Which observed levels of a signal move an outcome probability across a
    /// threshold.
    ThresholdCrossing {
        /// Outcome variable.
        outcome: String,
        /// Outcome level whose probability is tracked.
        outcome_level: String,
        /// The signal variable.
        signal: String,
        /// Recorded values held fixed.
        background: Vec<(String, String)>,
        /// Threshold, as a bit pattern.
        threshold_bits: u64,
        /// `raise` or `lower`.
        direction: String,
    },
    /// `P(outcome | do(interventions), observed values)` on a composed
    /// linear-Gaussian model: the interventional joint Gaussian conditioned on
    /// the observed real values by the Schur complement of the observed block.
    /// Either list may be empty. Format version 2.
    GaussianOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, value bits)` set by intervention.
        interventions: Vec<(String, u64)>,
        /// `(variable, value bits)` recorded.
        observed: Vec<(String, u64)>,
    },
    /// `P(outcome | do(interventions), observed values)` on a composed finite
    /// model: the truncated factorization conditioned on the recorded values,
    /// optionally with one measured signal inverted through its model. Format
    /// version 2.
    ConditionalInterventionalOutcome {
        /// Outcome variable.
        outcome: String,
        /// `(variable, level)` set by intervention.
        interventions: Vec<(String, String)>,
        /// `(variable, level)` recorded without error.
        observed: Vec<(String, String)>,
        /// One recorded indicator value, applied through its model.
        measured: Option<MeasuredSlotWire>,
    },
    /// The parameter uncertainty of a conditional functional by retained-data
    /// redraw: aligned posterior draws of the queried quantities. Format
    /// version 2.
    ParameterRedraw {
        /// The functional redrawn.
        target: RedrawTargetWire,
        /// The data the redrawn mechanisms were fitted from.
        retained: Vec<RetainedMechanismWire>,
        /// The priors.
        prior: RedrawPriorWire,
        /// Number of posterior draws.
        draws: u32,
        /// Seed of the draw streams.
        seed: u64,
    },
    /// Posterior of a declared finite hypothesis space after observations.
    FinitePosterior {
        /// Hypothesis names.
        hypotheses: Vec<String>,
        /// Prior probabilities, as bit patterns.
        prior_bits: Vec<u64>,
        /// Observation level names.
        observation_levels: Vec<String>,
        /// Row-major `[hypothesis][observation]` likelihood, as bit patterns.
        likelihood_bits: Vec<u64>,
        /// The observations (exchangeable).
        observations: Vec<String>,
    },
    /// Posterior of a declared Beta-Bernoulli model.
    ConjugatePosterior {
        /// Prior first shape, as bits.
        alpha_bits: u64,
        /// Prior second shape, as bits.
        beta_bits: u64,
        /// Successes observed.
        successes: u64,
        /// Failures observed.
        failures: u64,
    },
    /// The Bayes factor that moves a probability to a level.
    RequiredBayesFactor {
        /// Starting probability, as bits.
        prior_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
    },
    /// Readings of a binary signal needed to move a probability to a level.
    RequiredObservationsBinarySignal {
        /// Starting probability of the hypothesis, as bits.
        prior_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
        /// `P(positive reading | hypothesis)`, as bits.
        rate_given_hypothesis_bits: u64,
        /// `P(positive reading | complement)`, as bits.
        rate_given_complement_bits: u64,
        /// Largest observation count the solve may return.
        max_observations: u64,
    },
    /// Observations needed to move a Beta-Bernoulli predictive probability to
    /// a level.
    RequiredObservationsConjugate {
        /// Prior first shape, as bits.
        alpha_bits: u64,
        /// Prior second shape, as bits.
        beta_bits: u64,
        /// Target level, as bits.
        level_bits: u64,
        /// `raise` or `lower`.
        direction: String,
        /// Largest observation count the solve may return.
        max_observations: u64,
    },
}

impl ObservationQueryWire {
    /// The operation this query performs; part of the identity.
    #[must_use]
    pub fn operation(&self) -> &'static str {
        match self {
            Self::GaussianOutcome { interventions, observed, .. } => {
                match (interventions.is_empty(), observed.is_empty()) {
                    (true, _) => "conditioning",
                    (false, true) => "intervention",
                    (false, false) => "conditional_intervention",
                }
            }
            Self::ConditionalInterventionalOutcome { .. } => "conditional_intervention",
            Self::ParameterRedraw { .. } => "parameter_redraw",
            Self::ConditionalOutcome { .. } => "conditioning",
            Self::InterventionalOutcome { .. } => "intervention",
            Self::ConditioningVersusIntervention { .. } => "conditioning_and_intervention",
            Self::ThresholdCrossing { .. } => "threshold_crossing",
            Self::FinitePosterior { .. } | Self::ConjugatePosterior { .. } => "posterior_update",
            Self::RequiredBayesFactor { .. }
            | Self::RequiredObservationsBinarySignal { .. }
            | Self::RequiredObservationsConjugate { .. } => "evidence_requirement",
        }
    }

    /// Whether the query runs on a composed causal model.
    #[must_use]
    pub const fn needs_composed_model(&self) -> bool {
        matches!(
            self,
            Self::ConditionalOutcome { .. }
                | Self::InterventionalOutcome { .. }
                | Self::ConditioningVersusIntervention { .. }
                | Self::ThresholdCrossing { .. }
                | Self::GaussianOutcome { .. }
                | Self::ConditionalInterventionalOutcome { .. }
                | Self::ParameterRedraw { .. }
        )
    }

    /// Whether the query is one of the general queries, carried in format
    /// version 2 under [`super::OBSERVATION_QUERY_GENERAL_FEATURE`].
    #[must_use]
    pub const fn is_general(&self) -> bool {
        matches!(
            self,
            Self::GaussianOutcome { .. }
                | Self::ConditionalInterventionalOutcome { .. }
                | Self::ParameterRedraw { .. }
        )
    }

    /// This query with every assignment list in canonical (variable, level)
    /// order, so two declarations that differ only in listing order have one
    /// wire form and one identity. Results do not depend on the order.
    #[must_use]
    pub fn canonical(mut self) -> Self {
        match &mut self {
            Self::ConditionalOutcome { observed, .. } => observed.sort(),
            Self::InterventionalOutcome { interventions, .. } => interventions.sort(),
            Self::ConditioningVersusIntervention { assignment, .. } => assignment.sort(),
            Self::ThresholdCrossing { background, .. } => background.sort(),
            Self::GaussianOutcome { interventions, observed, .. } => {
                interventions.sort();
                observed.sort();
            }
            Self::ConditionalInterventionalOutcome { interventions, observed, .. } => {
                interventions.sort();
                observed.sort();
            }
            Self::ParameterRedraw { target, retained, .. } => {
                match target {
                    RedrawTargetWire::Gaussian { interventions, observed, .. } => {
                        interventions.sort();
                        observed.sort();
                    }
                    RedrawTargetWire::Finite { interventions, observed, .. } => {
                        interventions.sort();
                        observed.sort();
                    }
                }
                for item in retained.iter_mut() {
                    if let RetainedMechanismWire::Table { rows, .. } = item {
                        for (configuration, counts) in rows.iter_mut() {
                            configuration.sort();
                            counts.sort();
                        }
                        rows.sort();
                    }
                }
                retained.sort_by(|a, b| a.variable().cmp(b.variable()));
            }
            _ => {}
        }
        self
    }
}

pub(super) fn level_bits(levels: &[(String, f64)]) -> Vec<(String, u64)> {
    levels.iter().map(|(l, p)| (l.clone(), p.to_bits())).collect()
}

/// The stored result, as IEEE bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationResultWire {
    /// A conditional distribution.
    Conditional {
        /// Always `conditioning`.
        operation: String,
        /// `P(observed values)` (with the recorded indicator value, when one
        /// is applied), as bits.
        evidence_probability_bits: u64,
        /// `(level, probability bits)` of the outcome.
        levels: Vec<(String, u64)>,
        /// `(level, probability bits)` of the construct given a measured
        /// signal; `None` when no measured signal was applied.
        construct_posterior: Option<Vec<(String, u64)>>,
    },
    /// An interventional distribution.
    Interventional {
        /// Always `intervention`.
        operation: String,
        /// `(level, probability bits)` of the outcome.
        levels: Vec<(String, u64)>,
    },
    /// Both laws of one assignment.
    Paired {
        /// `P(assignment)` under the observational law, as bits.
        conditioning_evidence_bits: u64,
        /// Conditioning: `(level, probability bits)`.
        conditioning: Vec<(String, u64)>,
        /// Intervention: `(level, probability bits)`.
        intervention: Vec<(String, u64)>,
        /// Largest absolute difference over the outcome's levels, as bits.
        max_abs_difference_bits: u64,
    },
    /// The observed levels that cross a threshold.
    ThresholdCrossing {
        /// Probability of observing a crossing level, as bits.
        crossing_probability_bits: u64,
        /// `(signal level, P(level) bits, conditional bits, crosses)`.
        rows: Vec<(String, u64, u64, bool)>,
    },
    /// The Gaussian conditional law of one outcome.
    GaussianConditional {
        /// `conditioning`, `intervention` or `conditional_intervention`.
        operation: String,
        /// Conditional mean, as bits.
        mean_bits: u64,
        /// Conditional variance, as bits.
        variance_bits: u64,
        /// Log density of the observed values under their marginal law, as
        /// bits; `None` when nothing was observed.
        evidence_log_density_bits: Option<u64>,
    },
    /// Aligned posterior draws of the queried quantities.
    ParameterDraws {
        /// Functional ids of the quantities, in column order.
        quantities: Vec<String>,
        /// Number of draws.
        draws: u32,
        /// The quantities under the composition's own numbers, as bits.
        point_bits: Vec<u64>,
        /// Draw-major values, as bits.
        values_bits: Vec<u64>,
        /// Alignment identity of the draws (the distribution artifact's).
        alignment_identity: String,
        /// Always `uncalibrated_spread`: the draws are exact posterior draws and
        /// make no coverage statement.
        reading: String,
    },
    /// A finite posterior.
    FinitePosterior {
        /// `(hypothesis, posterior bits)`.
        posterior: Vec<(String, u64)>,
        /// Log evidence, as bits.
        log_evidence_bits: u64,
        /// Evidence probability, as bits.
        evidence_probability_bits: u64,
        /// `(observation level, predictive bits)`.
        predictive: Vec<(String, u64)>,
    },
    /// A conjugate posterior.
    ConjugatePosterior {
        /// Posterior first shape, as bits.
        alpha_bits: u64,
        /// Posterior second shape, as bits.
        beta_bits: u64,
        /// Predictive probability of the next success, as bits.
        predictive_success_bits: u64,
    },
    /// A required Bayes factor.
    RequiredBayesFactor {
        /// The factor, as bits.
        factor_bits: u64,
        /// No evidence is needed.
        already_satisfied: bool,
    },
    /// A required observation count.
    RequiredObservations {
        /// Smallest count that reaches the level.
        observations: u64,
        /// The probability at that count, as bits.
        value_bits: u64,
        /// No observation is needed.
        already_satisfied: bool,
        /// Bayes factor of one reading, as bits, for a likelihood-ratio route.
        bayes_factor_bits: Option<u64>,
    },
}

/// How far a replayed number may sit from the stored one.
///
/// Parameter redraws and the log, exp and gamma functions of the posterior queries run on
/// the platform's math library, which differs between operating systems in the last
/// digits. An artifact sealed on one platform therefore replays within this tolerance, not
/// bit for bit; integers, labels, counts and flags still match exactly. Identical bit
/// patterns always match, so a same-platform replay stays exact, including non-finite
/// values.
const OBSERVATION_REPLAY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::LargerOrOne, ReplayNonFinite::SameBits);

fn bits_close(stored: u64, replayed: u64) -> bool {
    OBSERVATION_REPLAY.close(f64::from_bits(stored), f64::from_bits(replayed))
}

fn all_bits_close(stored: &[u64], replayed: &[u64]) -> bool {
    stored.len() == replayed.len() && stored.iter().zip(replayed).all(|(s, r)| bits_close(*s, *r))
}

fn optional_bits_close(stored: Option<u64>, replayed: Option<u64>) -> bool {
    match (stored, replayed) {
        (Some(s), Some(r)) => bits_close(s, r),
        (None, None) => true,
        _ => false,
    }
}

fn levels_close(stored: &[(String, u64)], replayed: &[(String, u64)]) -> bool {
    stored.len() == replayed.len()
        && stored.iter().zip(replayed).all(|(s, r)| s.0 == r.0 && bits_close(s.1, r.1))
}

fn optional_levels_close(
    stored: Option<&Vec<(String, u64)>>,
    replayed: Option<&Vec<(String, u64)>>,
) -> bool {
    match (stored, replayed) {
        (Some(s), Some(r)) => levels_close(s, r),
        (None, None) => true,
        _ => false,
    }
}

impl ObservationResultWire {
    /// Whether this stored result is the replay of `replayed`: the same variant, exactly
    /// equal discrete fields and numbers within [`OBSERVATION_REPLAY`].
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per result variant keeps the field-by-field comparison reviewable"
    )]
    pub(super) fn replays_as(&self, replayed: &Self) -> bool {
        match (self, replayed) {
            (
                Self::Conditional {
                    operation: so,
                    evidence_probability_bits: se,
                    levels: sl,
                    construct_posterior: sc,
                },
                Self::Conditional {
                    operation: ro,
                    evidence_probability_bits: re,
                    levels: rl,
                    construct_posterior: rc,
                },
            ) => {
                so == ro
                    && bits_close(*se, *re)
                    && levels_close(sl, rl)
                    && optional_levels_close(sc.as_ref(), rc.as_ref())
            }
            (
                Self::Interventional { operation: so, levels: sl },
                Self::Interventional { operation: ro, levels: rl },
            ) => so == ro && levels_close(sl, rl),
            (
                Self::Paired {
                    conditioning_evidence_bits: se,
                    conditioning: sc,
                    intervention: si,
                    max_abs_difference_bits: sm,
                },
                Self::Paired {
                    conditioning_evidence_bits: re,
                    conditioning: rc,
                    intervention: ri,
                    max_abs_difference_bits: rm,
                },
            ) => {
                bits_close(*se, *re)
                    && levels_close(sc, rc)
                    && levels_close(si, ri)
                    && bits_close(*sm, *rm)
            }
            (
                Self::ThresholdCrossing { crossing_probability_bits: sp, rows: sr },
                Self::ThresholdCrossing { crossing_probability_bits: rp, rows: rr },
            ) => {
                bits_close(*sp, *rp)
                    && sr.len() == rr.len()
                    && sr.iter().zip(rr).all(|(s, r)| {
                        s.0 == r.0 && bits_close(s.1, r.1) && bits_close(s.2, r.2) && s.3 == r.3
                    })
            }
            (
                Self::GaussianConditional {
                    operation: so,
                    mean_bits: sm,
                    variance_bits: sv,
                    evidence_log_density_bits: sd,
                },
                Self::GaussianConditional {
                    operation: ro,
                    mean_bits: rm,
                    variance_bits: rv,
                    evidence_log_density_bits: rd,
                },
            ) => {
                so == ro
                    && bits_close(*sm, *rm)
                    && bits_close(*sv, *rv)
                    && optional_bits_close(*sd, *rd)
            }
            (
                Self::ParameterDraws {
                    quantities: sq,
                    draws: sd,
                    point_bits: sp,
                    values_bits: sv,
                    alignment_identity: sa,
                    reading: sr,
                },
                Self::ParameterDraws {
                    quantities: rq,
                    draws: rd,
                    point_bits: rp,
                    values_bits: rv,
                    alignment_identity: ra,
                    reading: rr,
                },
            ) => {
                sq == rq
                    && sd == rd
                    && sa == ra
                    && sr == rr
                    && all_bits_close(sp, rp)
                    && all_bits_close(sv, rv)
            }
            (
                Self::FinitePosterior {
                    posterior: sp,
                    log_evidence_bits: sl,
                    evidence_probability_bits: se,
                    predictive: sd,
                },
                Self::FinitePosterior {
                    posterior: rp,
                    log_evidence_bits: rl,
                    evidence_probability_bits: re,
                    predictive: rd,
                },
            ) => {
                levels_close(sp, rp)
                    && bits_close(*sl, *rl)
                    && bits_close(*se, *re)
                    && levels_close(sd, rd)
            }
            (
                Self::ConjugatePosterior {
                    alpha_bits: sa,
                    beta_bits: sb,
                    predictive_success_bits: sp,
                },
                Self::ConjugatePosterior {
                    alpha_bits: ra,
                    beta_bits: rb,
                    predictive_success_bits: rp,
                },
            ) => bits_close(*sa, *ra) && bits_close(*sb, *rb) && bits_close(*sp, *rp),
            (
                Self::RequiredBayesFactor { factor_bits: sf, already_satisfied: ss },
                Self::RequiredBayesFactor { factor_bits: rf, already_satisfied: rs },
            ) => ss == rs && bits_close(*sf, *rf),
            (
                Self::RequiredObservations {
                    observations: so,
                    value_bits: sv,
                    already_satisfied: ss,
                    bayes_factor_bits: sb,
                },
                Self::RequiredObservations {
                    observations: ro,
                    value_bits: rv,
                    already_satisfied: rs,
                    bayes_factor_bits: rb,
                },
            ) => so == ro && ss == rs && bits_close(*sv, *rv) && optional_bits_close(*sb, *rb),
            _ => false,
        }
    }
}
