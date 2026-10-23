// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter states of temporal intervention levels and temporal derivatives.
//!
//! Every check runs on a Pulse witness: the unit contrast at one horizon (with an
//! intervention level's own intervention as the active arm), identified by the same
//! horizon-specific proof, estimand and lag indexer the published value used. The
//! TemporalDag Pulse-native suite runs on each witness (overlap, the temporal placebo
//! and pre-period checks, and under `full` the circular-block bootstrap stability under
//! the declared dependent-series design), exactly as the graph-posterior temporal cells
//! run it on each atom. Bayesian studies fit each witness's posterior and add the prior
//! and posterior predictive checks and, under `full`, prior sensitivity.
//!
//! * A TemporalDag level has one witness per requested horizon.
//! * A TemporalCpdag / TemporalPag level has one witness per identified completion per
//!   horizon; completions are not mixed, and a per-horizon disagreement diagnostic
//!   reports how far the completions' witness effects spread.
//! * A temporal derivative (TemporalDag or class) has one witness per completion at its
//!   one horizon. Its state adds `derivative.support` and `derivative.finite_difference`
//!   (the unit-step difference of the identified levels — the weight-mixed witness
//!   effect — against the published identity-scale slope).
//!
//! Every state executes and is unlicensed: no coverage or evidence record names it.

use super::{
    AverageEffectQuery, BayesianGCompWorkspace, BayesianGComputationAte, BayesianValidation,
    CausalError, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity, EnvelopeAtomFit,
    EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId, ExecutionContext,
    IdentificationStatus, IdentifiedEstimand, InferenceMode, OverlapPolicy, PredictiveCheckReport,
    ResponseFunctional, ResponseQuery, TabularData, TemporalAtomDesign, TemporalEffectQuery,
    TemporalIndexer, TemporalLinearAdjustment, TemporalRefutation, TimeSeriesData, ValidationSink,
    bayesian_temporal_gcomp, envelope_refute_atoms, run_envelope_bayesian_full_validation,
};

/// Reports, predictive checks and notes of a temporal refuter state.
pub(super) type TemporalLevelRefutation =
    (Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>, Vec<Diagnostic>);

/// One Pulse witness: a horizon's (and, on a class, a completion's) identified
/// unit contrast with the products the published value used.
pub(super) struct PulseWitnessSite {
    /// Horizon-and-completion key (the class completion key, or the horizon).
    pub(super) key: u64,
    pub(super) horizon: u32,
    /// Completion mass (one on a TemporalDag).
    pub(super) weight: f64,
    pub(super) witness: TemporalEffectQuery,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) indexer: TemporalIndexer,
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: antecedent_core::AssumptionSet,
}

/// The witness of an intervention level at `horizon`: the unit contrast with the
/// level's intervention as the active arm.
pub(super) fn level_witness(
    query: &ResponseQuery,
    horizon: u32,
) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal refutation requires a TemporalResponseSpec".into(),
    })?;
    let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
    else {
        return Err(CausalError::Compile {
            message: "temporal level refuter state applies to an InterventionResponse".into(),
        });
    };
    let active = interventions.first().cloned().ok_or_else(|| CausalError::Compile {
        message: "InterventionResponse has no intervention coordinate".into(),
    })?;
    let treatment = active.primary_variable().ok_or_else(|| CausalError::Compile {
        message: "temporal refutation requires an intervention coordinate".into(),
    })?;
    Ok(TemporalEffectQuery {
        active,
        ..super::temporal_path::unit_contrast_query(
            temporal,
            (treatment, *outcome),
            horizon,
            query.target_population.clone(),
        )
    })
}

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
    let sites = horizons
        .iter()
        .map(|entry| {
            Ok(PulseWitnessSite {
                key: u64::from(entry.horizon),
                horizon: entry.horizon,
                weight: 1.0,
                witness: level_witness(query, entry.horizon)?,
                estimand: entry.estimand.clone(),
                indexer: entry.indexer.clone(),
                status: entry.identification.status,
                assumptions: entry.identification.required_assumptions.clone(),
            })
        })
        .collect::<Result<Vec<_>, CausalError>>()?;
    let (reports, checks, mut notes, _) = refute_pulse_witness_sites(study, data, &sites, ctx)?;
    notes.push(state_note("intervention-level", sites.len()));
    Ok((reports, checks, notes))
}

/// Run the per-completion, per-horizon refuter state of a TemporalCpdag / TemporalPag
/// intervention level.
///
/// # Errors
///
/// As [`refute_pulse_witness_sites`].
pub(super) fn refute_temporal_class_intervention(
    study: &super::Study,
    data: &TimeSeriesData,
    sites: &[PulseWitnessSite],
    ctx: &ExecutionContext,
) -> Result<TemporalLevelRefutation, CausalError> {
    let (reports, checks, mut notes, effects) =
        refute_pulse_witness_sites(study, data, sites, ctx)?;
    let mut horizons: Vec<u32> = sites.iter().map(|site| site.horizon).collect();
    horizons.sort_unstable();
    horizons.dedup();
    for horizon in horizons {
        let values: Vec<f64> = sites
            .iter()
            .zip(&effects)
            .filter(|(site, effect)| site.horizon == horizon && effect.is_finite())
            .map(|(_, effect)| *effect)
            .collect();
        notes.push(disagreement_note(horizon, &values));
    }
    notes.push(state_note("class intervention-level", sites.len()));
    Ok((reports, checks, notes))
}

/// Run the refuter state of a temporal derivative (TemporalDag or class).
///
/// # Errors
///
/// As [`refute_pulse_witness_sites`].
pub(super) fn refute_temporal_derivative(
    study: &super::Study,
    data: &TimeSeriesData,
    query: &ResponseQuery,
    response: &CausalResponse,
    sites: &[PulseWitnessSite],
    ctx: &ExecutionContext,
) -> Result<TemporalLevelRefutation, CausalError> {
    let (value, se) = super::derivative_refutation::derivative_scalar(response);
    let mut reports = vec![super::derivative_refutation::support_report(response, value)];
    let (witness_reports, checks, mut notes, effects) =
        refute_pulse_witness_sites(study, data, sites, ctx)?;
    let identity_scale = match &query.functional {
        ResponseFunctional::PointDerivative { order, scale, .. } => {
            *order == 1 && *scale == antecedent_core::DerivativeScale::Identity
        }
        ResponseFunctional::Jacobian { scale, .. } => {
            *scale == antecedent_core::DerivativeScale::Identity
        }
        ResponseFunctional::AverageDerivative { .. } => true,
        _ => false,
    };
    let (mass, weighted) = sites
        .iter()
        .zip(&effects)
        .filter(|(_, effect)| effect.is_finite())
        .fold((0.0, 0.0), |(mass, sum), (site, effect)| {
            (mass + site.weight, sum + site.weight * effect)
        });
    if identity_scale && mass > 0.0 {
        reports.push(super::derivative_refutation::finite_difference_report(
            value,
            se,
            weighted / mass,
        ));
    } else {
        notes.push(Diagnostic::new(
            "refute.derivative.finite_difference_not_applicable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the temporal finite-difference check compares an identity-scale first derivative \
             with the unit-step difference of the identified levels; this derivative is on \
             another scale, of another order, or has no evaluable witness",
        ));
    }
    reports.extend(witness_reports);
    notes.push(state_note("derivative", sites.len()));
    Ok((reports, checks, notes))
}

fn state_note(target: &str, witnesses: usize) -> Diagnostic {
    Diagnostic::new(
        "refute.temporal_response.pulse_witness_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "temporal {target} refuter state: the TemporalDag Pulse-native suite on each of \
             {witnesses} Pulse witness(es) (a horizon's, and on a class a completion's, \
             identified unit contrast under its own proof and lag indexer); witnesses are not \
             mixed. It executes unlicensed: no coverage or evidence record names it, and the \
             verdicts are checks, not a calibration claim."
        ),
    )
}

fn disagreement_note(horizon: u32, values: &[f64]) -> Diagnostic {
    let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = if values.is_empty() { f64::NAN } else { upper - lower };
    let agree = spread.is_finite() && spread <= 1e-9 * lower.abs().max(upper.abs()).max(1.0);
    Diagnostic::new(
        "refute.temporal_response.class_envelope_disagreement",
        DiagnosticKind::Scientific,
        if agree { DiagnosticSeverity::Info } else { DiagnosticSeverity::Warning },
        format!(
            "horizon {horizon}: {} completion witness effect(s) span [{lower}, {upper}] \
             (spread {spread}); each completion's refuter reports are kept on their own",
            values.len()
        ),
    )
    .with_fields([
        ("horizon", horizon.to_string()),
        ("completions", values.len().to_string()),
        ("lower", lower.to_string()),
        ("upper", upper.to_string()),
    ])
}

/// Run the Pulse-native suite on every witness site, one site at a time.
///
/// Returns the reports, predictive checks, notes, and each site's own witness effect
/// (the point for Frequentist inference, the posterior mean for Bayesian).
///
/// # Errors
///
/// A witness that cannot be prepared, cancellation, or a validator failure.
pub(super) fn refute_pulse_witness_sites(
    study: &super::Study,
    data: &TimeSeriesData,
    sites: &[PulseWitnessSite],
    ctx: &ExecutionContext,
) -> Result<
    (
        Vec<antecedent_validate::RefutationReport>,
        Vec<PredictiveCheckReport>,
        Vec<Diagnostic>,
        Vec<f64>,
    ),
    CausalError,
> {
    let tabular = TabularData::new(data.storage().clone());
    let mut reports = Vec::new();
    let mut checks = Vec::new();
    let mut notes = Vec::new();
    let mut effects = Vec::with_capacity(sites.len());
    for site in sites {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let witness = &site.witness;
        let ate_query = AverageEffectQuery::binary_ate(witness.treatment, witness.outcome);
        let frame = || TemporalRefutation {
            query: witness,
            split: study.split.as_ref(),
            time_index: data.time_index(),
            gaps: false,
        };
        match &study.inference {
            InferenceMode::Frequentist => {
                let design = TemporalAtomDesign::linear(
                    data,
                    &site.estimand,
                    witness,
                    &site.indexer,
                    study.split.as_ref(),
                    ctx,
                )?;
                let original = design.effect_estimate(site.assumptions.clone());
                effects.push(original.ate);
                let atom = EnvelopeRefuteAtom {
                    key: site.key,
                    weight: 1.0,
                    estimand: site.estimand.clone(),
                    indexer: Some(site.indexer.clone()),
                    original,
                };
                let (site_reports, site_notes) = EnvelopeRefutation {
                    data: &tabular,
                    query: &ate_query,
                    suite: study.refute,
                    estimator: "temporal.linear.adjustment",
                    custom: &[],
                    temporal: Some(frame()),
                }
                .run(&[atom], &mut EstimationWorkspace::default(), ctx)?;
                reports.extend(site_reports);
                notes.extend(site_notes);
            }
            InferenceMode::Bayesian(cfg) => {
                let bayes = bayesian_temporal_gcomp(cfg, ctx);
                let mut estimator = TemporalLinearAdjustment::new();
                estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
                let prep = estimator
                    .prepare(
                        data,
                        &site.estimand,
                        witness,
                        &site.indexer,
                        study.split.as_ref(),
                        &ctx.kernel_policy,
                    )
                    .map_err(CausalError::from)?;
                let names = antecedent_estimate::temporal_coefficient_names(
                    data,
                    &site.estimand,
                    witness,
                    &site.indexer,
                )
                .map_err(CausalError::from)?;
                let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                    .map_err(CausalError::from)?;
                let mut ws = BayesianGCompWorkspace::default();
                let posterior = bayes.fit(&bprep, site.status, &mut ws, ctx)?;
                let fits = vec![EnvelopeAtomFit {
                    key: site.key,
                    prep: bprep,
                    posterior: posterior.clone(),
                    status: site.status,
                    weight: 1.0,
                    estimand: site.estimand.clone(),
                    indexer: Some(site.indexer.clone()),
                    prior: bayes.inner.prior.clone(),
                }];
                let (mut site_reports, mut site_notes) = EnvelopeRefutation {
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
                effects.push(mean);
                let site_checks = run_envelope_bayesian_full_validation(
                    BayesianValidation { suite: study.refute, cfg, estimator: &bayes.inner },
                    &fits,
                    &mut mixture,
                    mean,
                    ctx,
                    super::super::latency::predictive_check_sims(study.latency_mode),
                    ValidationSink { refutations: &mut site_reports, diagnostics: &mut site_notes },
                )?;
                reports.extend(site_reports);
                notes.extend(site_notes);
                checks.extend(site_checks);
            }
        }
    }
    Ok((reports, checks, notes, effects))
}
