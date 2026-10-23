// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of an explicit or accepted `TemporalDag` intervention level.
//!
//! Every requested horizon is checked on its own Pulse witness: the unit contrast at
//! that horizon with the response's intervention as the active arm, identified by the
//! same horizon-specific temporal backdoor proof and lag indexer the response used. The
//! TemporalDag Pulse-native suite runs on that witness (overlap, the temporal placebo
//! and pre-period checks, and under `full` the circular-block bootstrap stability under
//! the declared dependent-series design), exactly as the graph-posterior TemporalDag cells
//! run it on each atom. Bayesian studies fit each witness's posterior and add the prior
//! and posterior predictive checks and, under `full`, prior sensitivity. Horizons are not
//! mixed: each horizon's reports are kept. The state executes and is unlicensed.

use super::{
    AverageEffectQuery, BayesianGCompWorkspace, BayesianGComputationAte, BayesianValidation,
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, EnvelopeAtomFit,
    EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId, ExecutionContext,
    InferenceMode, OverlapPolicy, PredictiveCheckReport, ResponseFunctional, ResponseQuery,
    TabularData, TemporalAtomDesign, TemporalEffectQuery, TemporalLinearAdjustment,
    TemporalRefutation, TimeSeriesData, ValidationSink, bayesian_temporal_gcomp,
    envelope_refute_atoms, run_envelope_bayesian_full_validation,
};

/// Reports, predictive checks and notes of the temporal intervention-level state.
pub(super) type TemporalLevelRefutation =
    (Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>, Vec<Diagnostic>);

/// Run the per-horizon Pulse-witness refuter state of a TemporalDag intervention level.
///
/// # Errors
///
/// A query that is not a temporal intervention response, a witness that cannot be
/// prepared, cancellation, or a validator failure.
pub(super) fn refute_temporal_intervention(
    study: &super::Study,
    data: &TimeSeriesData,
    query: &ResponseQuery,
    horizons: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    ctx: &ExecutionContext,
) -> Result<TemporalLevelRefutation, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal intervention refutation requires a TemporalResponseSpec".into(),
    })?;
    let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
    else {
        return Err(CausalError::Compile {
            message: "temporal refuter state applies to an InterventionResponse".into(),
        });
    };
    let active = interventions.first().cloned().ok_or_else(|| CausalError::Compile {
        message: "InterventionResponse has no intervention coordinate".into(),
    })?;
    let treatment = active.primary_variable().ok_or_else(|| CausalError::Compile {
        message: "temporal intervention refutation requires an intervention coordinate".into(),
    })?;
    let tabular = TabularData::new(data.storage().clone());
    let mut reports = Vec::new();
    let mut checks = Vec::new();
    let mut notes = Vec::new();
    for entry in horizons {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let witness = TemporalEffectQuery {
            active: active.clone(),
            ..super::temporal_path::unit_contrast_query(
                temporal,
                (treatment, *outcome),
                entry.horizon,
                query.target_population.clone(),
            )
        };
        let ate_query = AverageEffectQuery::binary_ate(witness.treatment, witness.outcome);
        let frame = || TemporalRefutation {
            query: &witness,
            split: study.split.as_ref(),
            time_index: data.time_index(),
            gaps: false,
        };
        let (horizon_reports, horizon_checks, horizon_notes) = match &study.inference {
            InferenceMode::Frequentist => {
                let design = TemporalAtomDesign::linear(
                    data,
                    &entry.estimand,
                    &witness,
                    &entry.indexer,
                    study.split.as_ref(),
                    ctx,
                )?;
                let atom = EnvelopeRefuteAtom {
                    key: u64::from(entry.horizon),
                    weight: 1.0,
                    estimand: entry.estimand.clone(),
                    indexer: Some(entry.indexer.clone()),
                    original: design
                        .effect_estimate(entry.identification.required_assumptions.clone()),
                };
                let (reports, notes) = EnvelopeRefutation {
                    data: &tabular,
                    query: &ate_query,
                    suite: study.refute,
                    estimator: "temporal.linear.adjustment",
                    custom: &[],
                    temporal: Some(frame()),
                }
                .run(&[atom], &mut EstimationWorkspace::default(), ctx)?;
                (reports, Vec::new(), notes)
            }
            InferenceMode::Bayesian(cfg) => {
                let bayes = bayesian_temporal_gcomp(cfg, ctx);
                let mut estimator = TemporalLinearAdjustment::new();
                estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
                let prep = estimator
                    .prepare(
                        data,
                        &entry.estimand,
                        &witness,
                        &entry.indexer,
                        study.split.as_ref(),
                        &ctx.kernel_policy,
                    )
                    .map_err(CausalError::from)?;
                let names = antecedent_estimate::temporal_coefficient_names(
                    data,
                    &entry.estimand,
                    &witness,
                    &entry.indexer,
                )
                .map_err(CausalError::from)?;
                let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                    .map_err(CausalError::from)?;
                let mut ws = BayesianGCompWorkspace::default();
                let posterior = bayes.fit(&bprep, entry.identification.status, &mut ws, ctx)?;
                let fits = vec![EnvelopeAtomFit {
                    key: u64::from(entry.horizon),
                    prep: bprep,
                    posterior: posterior.clone(),
                    status: entry.identification.status,
                    weight: 1.0,
                    estimand: entry.estimand.clone(),
                    indexer: Some(entry.indexer.clone()),
                    prior: bayes.inner.prior.clone(),
                }];
                let (mut reports, mut notes) = EnvelopeRefutation {
                    data: &tabular,
                    query: &ate_query,
                    suite: study.refute,
                    estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
                    custom: &[],
                    temporal: Some(frame()),
                }
                .run(
                    &envelope_refute_atoms(&fits)?,
                    &mut EstimationWorkspace::default(),
                    ctx,
                )?;
                let mut mixture = posterior;
                let mean = mixture
                    .effect_column()
                    .and_then(|column| mixture.summaries.mean.get(column).copied())
                    .unwrap_or(f64::NAN);
                let checks = run_envelope_bayesian_full_validation(
                    BayesianValidation { suite: study.refute, cfg, estimator: &bayes.inner },
                    &fits,
                    &mut mixture,
                    mean,
                    ctx,
                    super::super::latency::predictive_check_sims(study.latency_mode),
                    ValidationSink { refutations: &mut reports, diagnostics: &mut notes },
                )?;
                (reports, checks, notes)
            }
        };
        reports.extend(horizon_reports);
        checks.extend(horizon_checks);
        notes.extend(horizon_notes);
    }
    notes.push(Diagnostic::new(
        "refute.temporal_response.pulse_witness_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "temporal intervention-level refuter state: the TemporalDag Pulse-native suite on \
             each of {} horizon(s)' own Pulse witness (the unit contrast at that horizon with \
             the response's intervention as the active arm, under the horizon's own proof and \
             lag indexer); horizons are not mixed. It executes unlicensed: no coverage or \
             evidence record names it, and the verdicts are checks, not a calibration claim.",
            horizons.len()
        ),
    ));
    Ok((reports, checks, notes))
}
