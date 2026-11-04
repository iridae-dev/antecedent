//! Measured intervals of adaptive-experiment results.
//!
//! The original `adaptive_inference_result_v1` artifact keeps its point-only
//! standing. This adapter replays that artifact, recognizes the exact
//! procedure cells whose whole-procedure coverage was measured, builds the
//! measured interval with the construction the record describes, and resolves
//! it against the current coverage records. Any other procedure, prior,
//! threshold, exploration mass or outcome structure still reports its interval;
//! the calibration slot then says `not_measured` or `outside_measured_scope`,
//! because a record for one cell never covers another. Calibration annotates the
//! interval and never gates it.
//!
//! Supported cells:
//!
//! * Thompson sampling with a uniform Beta prior, two arms, a posterior
//!   probability stopping rule at a measured threshold and two looks: the
//!   equal-tailed credible interval of the second arm's success probability.
//! * Epsilon-greedy repeated measures with one final analysis: the Wald
//!   interval of the adaptively weighted contrast.
//! * Epsilon-greedy with time-uniform confidence sequences: the sequence's
//!   interval of the second arm's mean or of the contrast, valid at every look.
//! * Fixed randomization with group-sequential monitoring: the repeated
//!   confidence interval of the effect at the last performed look.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use crate::adaptive_inference_result_artifact::{
    AdaptiveInferenceResultArtifact, AdaptiveResultSpec,
};
use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AssignmentFamily, BoundarySidedness, DelayMechanism,
    MonitoringStrategy, OutcomeStructure, SpendingKind, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::adaptive_measured_interval as built;

/// A replayed result that carries no interval of the asked kind. A procedure outside the
/// measured cells is never refused: its interval is reported with `outside_measured_scope` or
/// `not_measured`.
fn refusal(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("not_supported"),
        message: format!("measured_interval.no_interval_of_this_kind: {message}"),
    }
}

/// The planned horizon when the interim analyses determine it, otherwise the units analysed.
fn horizon(contract: &AdaptiveDesignContract, look: usize) -> Result<u64, IoError> {
    if let Ok(planned) = planned_horizon(contract, look) {
        return Ok(planned);
    }
    let units = contract.analysis_units(look).map_err(|r| IoError::Refused {
        code: r.code,
        message: format!("{}: {}", r.detail, r.remedy.unwrap_or("")),
    })?;
    Ok(units.len() as u64)
}

fn bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Tag a basis whose procedure is not the measured one. The estimator then names a construction no
/// record measured, so the calibration slot reports `not_measured`; the interval is still reported.
fn outside_protocol(
    mut basis: antecedent_core::CalibrationBasis,
) -> antecedent_core::CalibrationBasis {
    basis.estimator = format!("{}+outside_measured_protocol", basis.estimator).into();
    basis
}

fn prefix_fractions_measured(contract: &AdaptiveDesignContract, expected: &[f64]) -> bool {
    contract.declaration().interim_analyses.as_ref().is_some_and(|looks| {
        !looks.is_empty()
            && looks.len() <= expected.len()
            && looks.iter().zip(expected).all(|(l, e)| bits(l.information_fraction, *e))
    })
}

fn immediate(contract: &AdaptiveDesignContract) -> bool {
    contract
        .declaration()
        .availability
        .as_ref()
        .is_some_and(|a| a.delay == DelayMechanism::Immediate)
}

fn min_probability_at_least(contract: &AdaptiveDesignContract, floor: f64) -> bool {
    contract.declaration().assignment_probabilities.as_ref().is_some_and(|rows| {
        rows.iter().flatten().copied().fold(f64::INFINITY, f64::min) + 1e-12 >= floor
    })
}

fn tagged(
    measured_protocol: bool,
    basis: antecedent_core::CalibrationBasis,
) -> antecedent_core::CalibrationBasis {
    if measured_protocol { basis } else { outside_protocol(basis) }
}

/// What the replayed contract and request resolve to.
struct Reading {
    cell: &'static str,
    scalar: &'static str,
    point: f64,
    interval: (f64, f64),
    basis: antecedent_core::CalibrationBasis,
    scope: serde_json::Value,
}

fn planned_horizon(contract: &AdaptiveDesignContract, look: usize) -> Result<u64, IoError> {
    let looks = contract
        .declaration()
        .interim_analyses
        .as_ref()
        .ok_or_else(|| refusal("the contract keeps no interim analyses"))?;
    let at = looks.get(look).ok_or_else(|| refusal("the look is not an interim analysis"))?;
    let planned = (f64::from(at.units_enrolled) / at.information_fraction).round();
    if !(planned.is_finite() && planned >= 1.0) {
        return Err(refusal("the planned horizon is not determined by the interim analyses"));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite positive rounded unit count bounded by the contract's unit limit"
    )]
    let planned = planned as u64;
    Ok(planned)
}

fn unit_mean(contract: &AdaptiveDesignContract, unit: usize) -> f64 {
    let m = &contract.declaration().outcomes[unit];
    m.iter().sum::<f64>() / m.len() as f64
}

fn arm_outcomes(
    contract: &AdaptiveDesignContract,
    look: usize,
    arm: u32,
) -> Result<Vec<f64>, IoError> {
    let declaration = contract.declaration();
    Ok(contract
        .analysis_units(look)
        .map_err(|r| IoError::Refused {
            code: r.code,
            message: format!("{}: {}", r.detail, r.remedy.unwrap_or("")),
        })?
        .into_iter()
        .filter(|&u| declaration.assigned_arm[u] == arm)
        .map(|u| unit_mean(contract, u))
        .collect())
}

fn thompson(
    contract: &AdaptiveDesignContract,
    spec: &AdaptiveResultSpec,
    values: &[f64],
    look: usize,
) -> Result<Reading, IoError> {
    let cell = contract.cell();
    let AdaptiveResultSpec::BayesianPosterior { prior_alpha, prior_beta, .. } = spec else {
        return Err(refusal("not a posterior request"));
    };
    let threshold = match contract.declaration().stopping_rule.map(|r| r.spec) {
        Some(StoppingSpec::PosteriorProbabilityBest { threshold }) => Some(threshold),
        _ => None,
    };
    if contract.arms() != 2 {
        return Err(refusal(
            "the replayed posterior holds the second arm's Beta parameters only for a two-arm trial",
        ));
    }
    let measured_protocol = cell.assignment == AssignmentFamily::ThompsonBetaBernoulli
        && cell.outcome == OutcomeStructure::SingleOutcome
        && cell.monitoring == MonitoringStrategy::FixedInterimLooks
        && cell.stopping == StoppingRuleKind::Efficacy
        && bits(*prior_alpha, 1.0)
        && bits(*prior_beta, 1.0)
        && threshold.is_some_and(|t| built::THOMPSON_STOP_THRESHOLDS.iter().any(|m| bits(*m, t)))
        && look + 1 == contract.looks()
        && prefix_fractions_measured(contract, &[0.5, 1.0])
        && immediate(contract);
    let planned = horizon(contract, look)?;
    let (alpha, beta, mean) = (values[6], values[7], values[8]);
    let interval = built::beta_credible_interval(alpha, beta, built::MEASURED_LEVEL)
        .ok_or_else(|| refusal("the posterior interval is not defined"))?;
    Ok(Reading {
        cell: "thompson_beta_bernoulli_posterior_probability_stop",
        scalar: "treatment_success_probability",
        point: mean,
        interval,
        basis: tagged(
            measured_protocol,
            built::calibration_basis(
                built::THOMPSON_ESTIMATOR,
                "posterior_quantile",
                "Bayesian",
                built::THOMPSON_FUNCTIONAL,
                built::STOPPED_DEPENDENCE,
                planned,
            ),
        ),
        scope: serde_json::json!({
            "validation_design":"two_bernoulli_arms_truth_drawn_from_the_uniform_prior_thompson_posterior_probability_stop",
            "evidence_interpretation":"prior-predictive calibration of the credible interval under the declared uniform prior; at a fixed true value near equal arms the stopped interval under-covers (named boundary) and no pointwise frequentist coverage is claimed",
            "declared_model_assumptions":["Bernoulli outcomes","uniform Beta(1,1) prior","assignment depends only on observed history","outcomes available at enrollment"],
            "stopping_threshold":threshold,"planned_horizon":planned,"looks":contract.looks(),
            "reported_scalar":"treatment_success_probability"
        }),
    })
}

fn weighted(
    contract: &AdaptiveDesignContract,
    spec: &AdaptiveResultSpec,
    values: &[f64],
    look: usize,
) -> Result<Reading, IoError> {
    let cell = contract.cell();
    let AdaptiveResultSpec::AdaptiveWeighting {
        plus_arm,
        minus_arm,
        scheme,
        prior_mean,
        prior_strength,
        size_weighted,
        ..
    } = spec
    else {
        return Err(refusal("not a weighting request"));
    };
    if contract.arms() != 2 || *plus_arm != 1 || *minus_arm != Some(0) {
        return Err(refusal(
            "the replayed weighted result reports the second arm against the first of a two-arm trial",
        ));
    }
    let declaration = contract.declaration();
    let no_rule =
        declaration.stopping_rule.is_none_or(|r| matches!(r.spec, StoppingSpec::NoStopping));
    let measured_protocol = cell.assignment == AssignmentFamily::EpsilonGreedy
        && cell.outcome == OutcomeStructure::RepeatedMeasures
        && cell.monitoring == MonitoringStrategy::FinalAnalysisOnly
        && cell.stopping == StoppingRuleKind::NoStopping
        && no_rule
        && contract.looks() == 1
        && look == 0
        && scheme == "variance_stabilizing"
        && bits(*prior_mean, 0.5)
        && bits(*prior_strength, 1.0)
        && !*size_weighted
        && bits(declaration.probability_floor, built::WEIGHTED_PROBABILITY_FLOOR)
        && declaration.outcomes.iter().all(|m| m.len() == 3)
        && immediate(contract)
        && min_probability_at_least(contract, built::MEASURED_EXPLORATION / 2.0);
    let units = contract
        .analysis_units(look)
        .map_err(|r| IoError::Refused {
            code: r.code,
            message: format!("{}: {}", r.detail, r.remedy.unwrap_or("")),
        })?
        .len() as u64;
    let (estimate, variance) = (values[0], values[1]);
    let interval = built::weighted_wald_interval(estimate, variance, built::MEASURED_LEVEL)
        .ok_or_else(|| refusal("the weighted interval is not defined"))?;
    Ok(Reading {
        cell: "epsilon_greedy_repeated_measures_final_analysis_weighted_contrast",
        scalar: "treatment_minus_control",
        point: estimate,
        interval,
        basis: tagged(
            measured_protocol,
            built::calibration_basis(
                built::WEIGHTED_ESTIMATOR,
                "analytic_se",
                "Frequentist",
                built::CONTRAST_FUNCTIONAL,
                built::FINAL_DEPENDENCE,
                units,
            ),
        ),
        scope: serde_json::json!({
            "validation_design":"three_measurements_per_subject_random_intercept_epsilon_greedy_0.3",
            "evidence_interpretation":"finite validation grid for a final analysis without stopping; a stopping look, outcome delay or exploration below 0.3 is not covered",
            "declared_model_assumptions":["subject means are the units","assignment depends only on observed history","outcomes available at enrollment"],
            "units":units,"exploration_floor":built::MEASURED_EXPLORATION/2.0,
            "reported_scalar":"treatment_minus_control"
        }),
    })
}

fn sequence(
    contract: &AdaptiveDesignContract,
    spec: &AdaptiveResultSpec,
    values: &[f64],
    look: usize,
) -> Result<Reading, IoError> {
    let cell = contract.cell();
    let AdaptiveResultSpec::ConfidenceSequence { alpha, tuning_samples, contrast, .. } = spec
    else {
        return Err(refusal("not a confidence-sequence request"));
    };
    let contrast_asked = match contrast {
        Some((1, 0)) => true,
        None => false,
        Some(_) => {
            return Err(refusal(
                "the replayed sequence reports the second arm's mean or its contrast with the first",
            ));
        }
    };
    if contract.arms() != 2 {
        return Err(refusal("the replayed sequence holds two arms"));
    }
    let measured_protocol = cell.assignment == AssignmentFamily::EpsilonGreedy
        && cell.outcome == OutcomeStructure::SingleOutcome
        && cell.monitoring == MonitoringStrategy::FixedInterimLooks
        && cell.stopping == StoppingRuleKind::Efficacy
        && bits(*alpha, built::SEQUENCE_ALPHA)
        && bits(*tuning_samples, built::SEQUENCE_TUNING_SAMPLES)
        && look + 1 == contract.looks()
        && immediate(contract)
        && min_probability_at_least(contract, built::MEASURED_EXPLORATION / 2.0);
    let planned = horizon(contract, look)?;
    let (scalar, functional, point, interval) = if contrast_asked {
        (
            "treatment_minus_control",
            built::CONTRAST_FUNCTIONAL,
            values[5] - values[1],
            (values[8], values[9]),
        )
    } else {
        ("treatment_mean", built::ARM_MEAN_FUNCTIONAL, values[5], (values[6], values[7]))
    };
    if !(interval.0.is_finite() && interval.1.is_finite()) || interval.0 > interval.1 {
        return Err(refusal("the confidence sequence is empty or not finite"));
    }
    Ok(Reading {
        cell: "epsilon_greedy_time_uniform_confidence_sequence",
        scalar,
        point,
        interval,
        basis: {
            let mut basis = tagged(
                measured_protocol,
                built::calibration_basis(
                    built::SEQUENCE_ESTIMATOR,
                    "confidence_sequence",
                    "Frequentist",
                    functional,
                    built::TIME_UNIFORM_DEPENDENCE,
                    planned,
                ),
            );
            basis.level = 1.0 - *alpha;
            basis
        },
        scope: serde_json::json!({
            "validation_design":"two_bernoulli_arms_0.4_0.6_epsilon_greedy_0.3_ten_looks",
            "evidence_interpretation":"one-sided floor: the sequence is conservative and held at least nominal coverage at every measured look; it is valid at a data-dependent stopping look",
            "declared_model_assumptions":["outcomes bounded in [0, 1]","assignment depends only on observed history"],
            "planned_horizon":planned,"reported_scalar":scalar
        }),
    })
}

fn group_sequential(
    contract: &AdaptiveDesignContract,
    spec: &AdaptiveResultSpec,
    values: &[f64],
    look: usize,
) -> Result<Reading, IoError> {
    let cell = contract.cell();
    let AdaptiveResultSpec::GroupSequential { control_arm, treatment_arm, .. } = spec else {
        return Err(refusal("not a group-sequential request"));
    };
    if contract.arms() != 2 || *control_arm != 0 || *treatment_arm != 1 {
        return Err(refusal(
            "the repeated interval is built for the second arm against the first of a two-arm trial",
        ));
    }
    let declaration = contract.declaration();
    let rule_ok = matches!(
        declaration.stopping_rule.map(|r| r.spec),
        Some(StoppingSpec::AlphaSpending {
            spending: SpendingKind::OBrienFlemingType,
            total_alpha,
            sidedness: BoundarySidedness::TwoSidedSymmetric,
        }) if bits(total_alpha, 0.05)
    );
    let one_to_one = declaration
        .assignment_probabilities
        .as_ref()
        .is_some_and(|rows| rows.iter().flatten().all(|p| (p - 0.5).abs() <= 1e-12));
    let measured_protocol = cell.assignment == AssignmentFamily::FixedRandomization
        && cell.outcome == OutcomeStructure::SingleOutcome
        && cell.monitoring == MonitoringStrategy::GroupSequential
        && cell.stopping == StoppingRuleKind::Efficacy
        && rule_ok
        && look + 1 == contract.looks()
        && prefix_fractions_measured(contract, &[1.0 / 3.0, 2.0 / 3.0, 1.0])
        && immediate(contract)
        && one_to_one;
    let planned = horizon(contract, look)?;
    let (estimate, se) = built::welch_difference(
        &arm_outcomes(contract, look, 0)?,
        &arm_outcomes(contract, look, 1)?,
    )
    .ok_or_else(|| refusal("the arm difference and its standard error are not defined"))?;
    let critical = *values
        .get(1 + 3 * look)
        .ok_or_else(|| refusal("the result holds no boundary for the look"))?;
    let interval = built::repeated_confidence_interval(estimate, se, critical)
        .ok_or_else(|| refusal("the repeated interval is not defined"))?;
    Ok(Reading {
        cell: "fixed_randomization_group_sequential_repeated_interval",
        scalar: "treatment_minus_control",
        point: estimate,
        interval,
        basis: tagged(
            measured_protocol,
            built::calibration_basis(
                built::REPEATED_INTERVAL_ESTIMATOR,
                "repeated_confidence_interval",
                "Frequentist",
                built::CONTRAST_FUNCTIONAL,
                built::GROUP_SEQUENTIAL_DEPENDENCE,
                planned,
            ),
        ),
        scope: serde_json::json!({
            "validation_design":"fixed_randomization_three_looks_obrien_fleming_two_sided_normal_outcomes",
            "evidence_interpretation":"one-sided floor: the repeated interval is simultaneous over looks and conservative; it is read at the last performed look; stagewise-adjusted estimates and p-values stay closed",
            "declared_model_assumptions":["independent increments under fixed randomization","normal outcomes"],
            "planned_horizon":planned,"look":look,"boundary":critical,
            "reported_scalar":"treatment_minus_control"
        }),
    })
}

impl AdaptiveInferenceResultArtifact {
    /// Replay the original adaptive result and bind its interval to currently
    /// attesting measured coverage.
    ///
    /// # Errors
    /// Original replay failures, a procedure outside the measured cells, or
    /// absent or stale evidence.
    pub fn measure_interval(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let artifact = Self::from_bytes(bytes, None, ctx).map_err(|error| {
            error.refusal().map_or_else(
                || IoError::Refused {
                    code: antecedent_core::reason_code!("invalid_argument"),
                    message: format!("adaptive_inference.input_invalid: {error}"),
                },
                |(code, detail, message)| IoError::Refused {
                    code,
                    message: format!("{detail}: {message}"),
                },
            )
        })?;
        let contract = artifact.contract().contract();
        let spec = artifact.spec();
        let values = &artifact.result().values;
        let look = spec.look() as usize;
        let reading = match spec {
            AdaptiveResultSpec::BayesianPosterior { .. } => thompson(contract, spec, values, look)?,
            AdaptiveResultSpec::AdaptiveWeighting { .. } => weighted(contract, spec, values, look)?,
            AdaptiveResultSpec::ConfidenceSequence { .. } => {
                sequence(contract, spec, values, look)?
            }
            AdaptiveResultSpec::GroupSequential { .. } => {
                group_sequential(contract, spec, values, look)?
            }
            _ => {
                return Err(refusal("no measured interval exists for this inferential paradigm"));
            }
        };
        let premises =
            crate::digest_hex(IdentityDomain::Claim, &(reading.cell, artifact.identity()))?;
        let data = crate::identity::digest_canonical(IdentityDomain::Claim, bytes).to_hex();
        let mut scope = reading.scope;
        if let Some(map) = scope.as_object_mut() {
            map.insert("cell".into(), serde_json::Value::from(reading.cell));
        }
        authorize(
            "adaptive_interval",
            bytes.to_vec(),
            &premises,
            &data,
            scope,
            vec![ScalarExecution {
                name: reading.scalar.into(),
                point: reading.point,
                interval: reading.interval,
                basis: reading.basis,
            }],
        )
    }

    /// Consume a measured envelope by replaying the original adaptive result and
    /// re-resolving its scalar against current records.
    ///
    /// # Errors
    /// Any source, expected identity, protocol, evidence or scalar mismatch.
    pub fn consume_measured_interval(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let wire = decode(bytes, expected)?;
        let measured = Self::measure_interval(&wire.source, ctx)?;
        compare_replay(&wire, &measured)?;
        Ok(measured)
    }
}
