//! Measured intervals of choice-study results.
//!
//! The `choice_inference_result_v1` artifact keeps its point-only standing. This adapter replays
//! it, recognizes the exact choice cells whose coverage is measured, builds the interval with
//! the construction the record describes and resolves it against the current coverage records.
//! Any other contract, prior, route or outcome structure is refused: a record for one cell never
//! covers another.
//!
//! Cells (a record exists for none until coverage is measured; an unresolved key refuses):
//!
//! * categorical choice, one final analysis, no retained sources, uniform Dirichlet prior: the
//!   equal-tailed credible interval of the first option's share;
//! * repeated choice, one final analysis, uniform share shapes and the measured heterogeneity
//!   hyperprior: the equal-tailed credible interval of the first option's population share.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use crate::choice_inference_result_artifact::{ChoiceInferenceResultArtifact, ChoiceResultSpec};
use antecedent_contract::adaptive_design::{
    ChoiceDesignContract, MonitoringStrategy, OutcomeStructure, StoppingRuleKind,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::choice_inference as built;

/// The `ln kappa` hyperprior the hierarchical interval was measured under: mean `ln 5`, sd 1.
fn measured_log_concentration() -> (f64, f64) {
    (5f64.ln(), 1.0)
}

fn refusal(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("measured_interval.protocol_not_measured: {message}"),
    }
}

fn bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn final_only(contract: &ChoiceDesignContract, look: usize) -> Result<u64, IoError> {
    let cell = contract.cell();
    if cell.monitoring != MonitoringStrategy::FinalAnalysisOnly
        || cell.stopping != StoppingRuleKind::NoStopping
        || look + 1 != contract.looks()
    {
        return Err(refusal(
            "only one final analysis without a stopping rule was measured for choice studies",
        ));
    }
    Ok(contract.respondents_at(look).map_err(|_| refusal("the look is not declared"))?.len() as u64)
}

struct Reading {
    cell: &'static str,
    point: f64,
    interval: (f64, f64),
    basis: antecedent_core::CalibrationBasis,
    scope: serde_json::Value,
}

fn reading(
    contract: &ChoiceDesignContract,
    spec: &ChoiceResultSpec,
    values: &[f64],
    ctx: &ExecutionContext,
) -> Result<Reading, IoError> {
    let look = spec.look() as usize;
    match spec {
        ChoiceResultSpec::Posterior { .. } => {
            let declaration = contract.declaration();
            if contract.cell().outcome != OutcomeStructure::CategoricalChoice
                || !declaration.retained_sources.is_empty()
                || !declaration.initial_prior.iter().all(|p| bits(*p, 1.0))
            {
                return Err(refusal(
                    "only a categorical choice study on the uniform Dirichlet prior without retained sources was measured",
                ));
            }
            let planned = final_only(contract, look)?;
            let alpha: Vec<f64> = (0..contract.options()).map(|k| values[4 * k]).collect();
            let interval = built::dirichlet_share_interval(&alpha, 0, built::MEASURED_LEVEL)
                .map_err(|e| refusal(&e.message))?;
            Ok(Reading {
                cell: "dirichlet_multinomial_final_analysis",
                point: values[1],
                interval,
                basis: built::calibration_basis(
                    built::DIRICHLET_ESTIMATOR,
                    built::FIRST_OPTION_SHARE_FUNCTIONAL,
                    built::POOLED_DEPENDENCE,
                    built::DIRICHLET_PRIOR,
                    planned,
                ),
                scope: serde_json::json!({
                    "validation_design":"respondent shares drawn from the uniform Dirichlet prior, one choice per respondent, one final analysis",
                    "evidence_interpretation":"prior-predictive calibration of the credible interval under the declared uniform prior; no pointwise frequentist coverage is claimed and stated choices do not identify behaviour",
                    "declared_model_assumptions":["one choice per respondent","exchangeable respondents","uniform Dirichlet prior","no retained sources"],
                    "planned_respondents":planned,"options":contract.options(),
                    "reported_scalar":"first_option_share"
                }),
            })
        }
        ChoiceResultSpec::Hierarchical {
            shapes,
            log_concentration_mean,
            log_concentration_sd,
            ..
        } => {
            if !shapes.iter().all(|s| bits(*s, 1.0))
                || !bits(*log_concentration_mean, measured_log_concentration().0)
                || !bits(*log_concentration_sd, measured_log_concentration().1)
            {
                return Err(refusal(
                    "only uniform population-share shapes with ln kappa ~ Normal(ln 5, 1) were measured",
                ));
            }
            let planned = final_only(contract, look)?;
            let prior = built::HierarchicalChoicePrior {
                shapes: shapes.clone(),
                log_concentration_mean: *log_concentration_mean,
                log_concentration_sd: *log_concentration_sd,
            };
            let respondents: Vec<Vec<u32>> = contract
                .respondents_at(look)
                .map_err(|_| refusal("the look is not declared"))?
                .iter()
                .map(|r| r.counts.clone())
                .collect();
            let interval = built::hierarchical_share_interval(
                &prior,
                &respondents,
                0,
                built::MEASURED_LEVEL,
                ctx,
            )
            .map_err(|e| refusal(&e.message))?;
            Ok(Reading {
                cell: "hierarchical_dirichlet_multinomial_final_analysis",
                point: values[0],
                interval,
                basis: built::calibration_basis(
                    built::HIERARCHICAL_ESTIMATOR,
                    built::FIRST_OPTION_SHARE_FUNCTIONAL,
                    built::HIERARCHICAL_DEPENDENCE,
                    built::HIERARCHICAL_PRIOR,
                    planned,
                ),
                scope: serde_json::json!({
                    "validation_design":"population shares and respondent heterogeneity drawn from the declared hyperprior, repeated choices per respondent, one final analysis",
                    "evidence_interpretation":"prior-predictive calibration of the credible interval under the declared hyperprior; no pointwise frequentist coverage is claimed and stated choices do not identify behaviour",
                    "declared_model_assumptions":["exchangeable respondents","respondent shares Dirichlet around the population shares","uniform population-share shapes","ln kappa Normal(ln 5, 1)"],
                    "planned_respondents":planned,"options":contract.options(),
                    "reported_scalar":"first_option_population_share"
                }),
            })
        }
        _ => Err(refusal("no measured interval exists for a stopping decision")),
    }
}

impl ChoiceInferenceResultArtifact {
    /// Replay the original choice result and bind its interval to currently attesting measured
    /// coverage.
    ///
    /// # Errors
    /// Original replay failures, a study outside the measured cells, or absent or stale
    /// evidence.
    pub fn measure_interval(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let artifact = Self::from_bytes(bytes, None, ctx).map_err(|error| {
            error.refusal().map_or_else(
                || IoError::Refused {
                    code: antecedent_core::reason_code!("invalid_argument"),
                    message: format!("choice_inference.input_invalid: {error}"),
                },
                |(code, detail, message)| IoError::Refused {
                    code,
                    message: format!("{detail}: {message}"),
                },
            )
        })?;
        let reading =
            reading(artifact.contract(), artifact.spec(), &artifact.result().values, ctx)?;
        let premises =
            crate::digest_hex(IdentityDomain::Claim, &(reading.cell, artifact.identity()))?;
        let data = crate::identity::digest_canonical(IdentityDomain::Claim, bytes).to_hex();
        let mut scope = reading.scope;
        if let Some(map) = scope.as_object_mut() {
            map.insert("cell".into(), serde_json::Value::from(reading.cell));
        }
        authorize(
            "choice_interval",
            bytes.to_vec(),
            &premises,
            &data,
            scope,
            vec![ScalarExecution {
                name: scope_name(reading.cell).into(),
                point: reading.point,
                interval: reading.interval,
                basis: reading.basis,
            }],
        )
    }

    /// Consume a measured envelope by replaying the original choice result and re-resolving its
    /// scalar against current records.
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

fn scope_name(cell: &str) -> &'static str {
    if cell.starts_with("hierarchical") {
        "first_option_population_share"
    } else {
        "first_option_share"
    }
}
