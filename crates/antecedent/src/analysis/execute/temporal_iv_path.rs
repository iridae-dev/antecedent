// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lagged-instrument effect of a single-step temporal query on one series.
//!
//! The instrument is a lagged variable the unfolded window graph certifies: not a descendant of
//! the treatment, connected to it, and separated from the outcome once the treatment's out-edges
//! are cut. The graph supplies exclusion and exogeneity; a time column supplies neither.
//! Relevance is checked at every fit by the first-stage F statistic of the data, with a fixed
//! threshold of 10, and a weaker first stage is refused.
//!
//! The estimate is the Wald contrast of the lag-aligned rows: instrument at `t - lag`, treatment
//! at `t`, outcome `horizon` steps after the treatment. Rows of one series are serially dependent
//! and the contrast treats them as independent, so its homoskedastic standard error and the
//! Bayesian reduced-form ratio posterior carry no serial-dependence correction; that premise is
//! recorded on the result. Neither estimator is licensed and calibration is unmeasured.

use antecedent_estimate::DrawSpec;
use antecedent_estimate::lagged_instrument::{
    LaggedInstrumentSample, fit_lagged_instrument, fit_lagged_instrument_posterior,
};

use super::panel_link_path::effect_draws_posterior;
use super::{
    Arc, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId,
    ExecutionContext, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceMode, Instant, OverlapPolicy, PhysicalExecutionPlan, Strategy, StudyResult, TableView,
    TemporalDag, TemporalEffectQuery, effect_from_posterior, elapsed_ns,
};
use crate::analysis::temporal_iv_identification::identify_temporal_instrument;
use antecedent_data::TimeSeriesData;

/// First-stage F below which the lagged instrument is refused as weak.
const WEAK_F_THRESHOLD: f64 = 10.0;
const IV_BACKEND: &str = "gaussian.temporal_lagged_instrument_ratio";

impl super::Study {
    /// Execute the lagged-instrument estimator selected on a temporal effect.
    #[allow(
        clippy::too_many_lines,
        reason = "one estimator: certify the lag, fit the contrast or the reduced-form posterior, and record the independent-row premise"
    )]
    pub(super) fn execute_temporal_iv(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, indexer, certified_instrument, identify_cached) =
            if let Some((cache, prepared)) = self.reuse_temporal_identification(physical) {
                let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                    message: "prepared temporal identification missing query horizon".into(),
                })?;
                (
                    entry.identification.clone(),
                    entry.estimand.clone(),
                    entry.indexer.clone(),
                    None,
                    prepared,
                )
            } else {
                super::report_identify_compute(ctx);
                let found = identify_temporal_instrument(graph, query)?;
                (found.identification, found.estimand, found.indexer, Some(found.instrument), false)
            };
        let instrument = if let Some(instrument) = certified_instrument {
            instrument
        } else {
            let dense =
                estimand.instruments.first().copied().ok_or_else(|| CausalError::Compile {
                    message: "temporal instrument estimand names no instrument".into(),
                })?;
            indexer.key_of(dense.raw()).map_err(CausalError::rejected)?
        };
        let treatment_at = query.try_treatment_offset().map_err(CausalError::rejected)?;
        let lag = usize::try_from(treatment_at - instrument.offset).map_err(|_| {
            CausalError::Compile {
                message: "the temporal instrument does not precede the treatment".into(),
            }
        })?;
        let horizon = usize::try_from(query.outcome_offset() - treatment_at).map_err(|_| {
            CausalError::Compile { message: "the outcome precedes the treatment".into() }
        })?;
        let columns = (
            data.float64_values(instrument.variable).map_err(CausalError::rejected)?,
            data.float64_values(query.treatment).map_err(CausalError::rejected)?,
            data.float64_values(query.outcome).map_err(CausalError::rejected)?,
        );
        let kept = columns.0.len().saturating_sub(horizon);
        let (instrument_rows, treatment_rows, outcome_rows) = (
            columns.0.get(..kept).unwrap_or_default(),
            columns.1.get(..kept).unwrap_or_default(),
            columns.2.get(horizon..).unwrap_or_default(),
        );
        let sample = LaggedInstrumentSample {
            instrument: instrument_rows,
            treatment: treatment_rows,
            outcome: outcome_rows,
            lag,
        };
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(dependence_assumption());
        let (estimate, posterior, estimator_id, note, n_draws) = match &self.inference {
            InferenceMode::Frequentist => {
                let fit =
                    fit_lagged_instrument(&sample, WEAK_F_THRESHOLD).map_err(CausalError::from)?;
                let estimate = EffectEstimate::new(
                    fit.effect,
                    fit.homoskedastic_se,
                    assumptions,
                    OverlapPolicy::ExplicitOverride,
                )
                .with_n_obs(u64::try_from(fit.aligned_rows).unwrap_or(u64::MAX));
                let note = format!(
                    "{}; first-stage F {:.3}, {} aligned rows, lag {}; homoskedastic standard \\
                     error of independent rows; calibration unmeasured",
                    fit.model, fit.first_stage_f, fit.aligned_rows, fit.lag
                );
                (estimate, None, EstimatorId::TemporalLaggedInstrument, note, None)
            }
            InferenceMode::Bayesian(cfg) => {
                if cfg.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "bayesian.temporal.lagged_instrument uses independent Gaussian \\
                                  priors on the reduced-form coefficients and does not accept \\
                                  coefficient or transferred priors",
                    });
                }
                let fit = fit_lagged_instrument_posterior(
                    &sample,
                    cfg.prior_scale,
                    DrawSpec::new(cfg.n_draws, ctx.rng.master_seed()),
                    WEAK_F_THRESHOLD,
                )
                .map_err(CausalError::from)?;
                let posterior = effect_draws_posterior(
                    &fit.effect_draws,
                    identification.status,
                    assumptions,
                    IV_BACKEND,
                )?;
                let estimate = effect_from_posterior(&posterior)?
                    .with_n_obs(u64::try_from(fit.aligned_rows).unwrap_or(u64::MAX));
                let note = format!(
                    "{}; first-stage F {:.3}, {} aligned rows, lag {}; N(0, {}^2) prior on each \\
                     reduced-form coefficient; calibration unmeasured",
                    fit.model, fit.first_stage_f, fit.aligned_rows, fit.lag, fit.prior_sd
                );
                let draws = u32::try_from(fit.effect_draws.len()).unwrap_or(u32::MAX);
                (
                    estimate,
                    Some(posterior),
                    EstimatorId::BayesianTemporalLaggedInstrument,
                    note,
                    Some(draws),
                )
            }
        };
        let diagnostic = Diagnostic::new(
            "estimate.temporal.lagged_instrument",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            note,
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::TemporalIvUnfolded, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws,
                diagnostics: Some(vec![diagnostic]),
                estimate_provenance: Some((
                    Arc::from("estimate.temporal_lagged_instrument"),
                    Arc::from("estimate.temporal.lagged_instrument"),
                )),
                ..Default::default()
            }),
        ))
    }
}

fn dependence_assumption() -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("temporal.lagged_instrument.independent_rows"),
                description: Arc::from(
                    "the lag-aligned rows of one series are treated as independent: the standard \\
                     error or posterior carries no serial-dependence correction",
                ),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal.lagged_instrument"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}
