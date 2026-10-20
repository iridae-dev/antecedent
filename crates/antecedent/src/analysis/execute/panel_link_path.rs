// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mean-scale contrast of a Bernoulli temporal model over independent subjects of a panel.
//!
//! The pooled panel route stacks every unit's lag-aligned rows, which are serially dependent
//! within a unit. The link contrast needs independent rows, so each unit contributes exactly one
//! decision time: its earliest row with a full lag window. The treatment column of that row is
//! replaced by the active and then the control level while the lag columns stay at their observed
//! values, and the reported number is the average probability-scale difference, not the
//! link-scale treatment coefficient. Later rows of a unit do not enter; using them needs a
//! within-unit dependence correction this construction does not have.
//!
//! The frequentist estimate is the unpenalized maximum-likelihood fit with its Fisher delta-method
//! standard error. The Bayesian estimate is the Laplace posterior of the same contrast under a
//! Gaussian coefficient prior. Neither is a licensed route and their calibration is unmeasured.

use std::collections::BTreeMap;

use antecedent_estimate::temporal_link::{
    BayesianTemporalLink, FrequentistTemporalLink, RegimeColumn, TemporalContrastDesign,
    TemporalOutcomeLink, TrajectoryDependence,
};

use super::panel_path::{PanelPulseAtom, panel_unit_ids};
use super::{
    Arc, CausalError, CausalPosterior, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, InferenceDiagnostics, InferenceMode, Instant,
    OverlapPolicy, PanelData, PhysicalExecutionPlan, Strategy, StudyResult, TemporalDag,
    TemporalEffectQuery, effect_from_posterior, elapsed_ns, require_identified,
};

const LINK_BACKEND: &str = "laplace.temporal_glm_link_contrast";

impl super::Study {
    /// Execute the panel link contrast for the selected frequentist or Bayesian estimator.
    pub(super) fn execute_panel_link(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unlicensed_panel(panel)?;
        if query.is_multi_step_sustained() || !query.effect_modifiers.is_empty() {
            return Err(CausalError::Unsupported {
                message: "the temporal GLM link contrast is a single-step pulse or one-step \
                          sustained effect without effect modifiers",
            });
        }
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) =
            self.identify_pooled_panel(graph, query, ctx)?;
        require_identified(&identification)?;
        let unit_ids = panel_unit_ids(panel);
        let atom = PanelPulseAtom::prepare(
            panel,
            &estimand,
            query,
            &indexer,
            self.split.as_ref(),
            ctx,
            1.0,
        )?;
        let (prep, cluster_ids, panel_times) = atom.stacked(&unit_ids)?;
        let design =
            independent_subject_design(&subject_columns(&prep, &cluster_ids, &panel_times)?)?;
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(link_assumption(design_note(&design)));
        let (estimate, posterior, estimator_id, notes) = match &self.inference {
            InferenceMode::Frequentist => {
                let fit = FrequentistTemporalLink::new()
                    .fit(&design, TemporalOutcomeLink::BernoulliLogit)
                    .map_err(CausalError::from)?;
                let estimate = EffectEstimate::new(
                    fit.estimate,
                    fit.standard_error,
                    assumptions,
                    OverlapPolicy::ExplicitOverride,
                )
                .with_n_obs(u64::try_from(fit.trajectories).unwrap_or(u64::MAX));
                let note = format!(
                    "logit link, {} independent subjects, {} lag column(s) held at their observed \
                     values; Fisher delta-method standard error; calibration unmeasured",
                    fit.trajectories, fit.lags
                );
                (estimate, None, EstimatorId::TemporalGlmLinkContrast, note)
            }
            InferenceMode::Bayesian(cfg) => {
                let link = bayesian_link(cfg)?;
                let fit =
                    BayesianTemporalLink::new(cfg.prior_scale, cfg.n_draws, ctx.rng.master_seed())
                        .and_then(|bayes| bayes.fit(&design, link))
                        .map_err(CausalError::from)?;
                let posterior = effect_draws_posterior(
                    &fit.draws,
                    identification.status,
                    assumptions.clone(),
                    LINK_BACKEND,
                )?;
                let estimate = effect_from_posterior(&posterior)?
                    .with_n_obs(u64::try_from(fit.trajectories).unwrap_or(u64::MAX));
                let note = format!(
                    "{link:?} link, {} independent subjects, Laplace posterior under N(0, {}^2) \
                     on every coefficient; the posterior mean {:.6} is not the contrast at the mode \
                     {:.6}; calibration unmeasured",
                    fit.trajectories, cfg.prior_scale, fit.posterior_mean, fit.functional_at_mode
                );
                (estimate, Some(posterior), EstimatorId::BayesianTemporalGlmLinkContrast, note)
            }
        };
        let diagnostic = Diagnostic::new(
            "estimate.temporal.glm_link_contrast",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            notes,
        );
        let n_draws =
            posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX));
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::TemporalBackdoorUnfolded, estimator_id),
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
                    Arc::from("estimate.temporal_glm_link_contrast"),
                    Arc::from("estimate.temporal.glm_link_contrast"),
                )),
                ..Default::default()
            }),
        ))
    }
}

/// The link a Bayesian configuration declares through its likelihood.
fn bayesian_link(cfg: &crate::BayesianConfig) -> Result<TemporalOutcomeLink, CausalError> {
    if cfg.has_transfer() {
        return Err(CausalError::Unsupported {
            message: "bayesian.temporal.glm_link_contrast uses an isotropic Gaussian coefficient \
                      prior and does not accept coefficient or transferred priors",
        });
    }
    match cfg.likelihood {
        antecedent_prob::BayesLikelihood::BernoulliLogit => Ok(TemporalOutcomeLink::BernoulliLogit),
        antecedent_prob::BayesLikelihood::BernoulliProbit => {
            Ok(TemporalOutcomeLink::BernoulliProbit)
        }
        _ => Err(CausalError::Unsupported {
            message: "bayesian.temporal.glm_link_contrast needs a Bernoulli logit or probit \
                      likelihood on the Bayesian configuration",
        }),
    }
}

/// The design columns of the independent-subject rows of a panel pulse.
pub(crate) struct SubjectColumns {
    /// Subjects: one row per unit.
    pub(crate) nrows: usize,
    /// Design columns over those rows: intercept, treatment, then the adjustment lags.
    pub(crate) columns: Vec<Vec<f64>>,
    /// Outcome over those rows.
    pub(crate) outcome: Vec<f64>,
    /// Active treatment level.
    pub(crate) active: f64,
    /// Control treatment level.
    pub(crate) control: f64,
}

/// One row per unit: the earliest aligned row.
fn subject_columns(
    prep: &antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: &[u32],
    panel_times: &[i64],
) -> Result<SubjectColumns, CausalError> {
    let n = prep.design.nrows;
    if cluster_ids.len() != n || panel_times.len() != n {
        return Err(CausalError::Compile {
            message: "panel subject rows: unit ids and times do not align with the stacked rows"
                .into(),
        });
    }
    let mut earliest: BTreeMap<u32, (i64, usize)> = BTreeMap::new();
    for (row, (&unit, &time)) in cluster_ids.iter().zip(panel_times).enumerate() {
        earliest
            .entry(unit)
            .and_modify(|slot| {
                if time < slot.0 {
                    *slot = (time, row);
                }
            })
            .or_insert((time, row));
    }
    let mut rows: Vec<usize> = earliest.values().map(|&(_, row)| row).collect();
    rows.sort_unstable();
    let out_of_range =
        || CausalError::Compile { message: "panel subject rows: design index out of range".into() };
    let mut columns = Vec::with_capacity(prep.design.ncols);
    for column in 0..prep.design.ncols {
        let values = rows
            .iter()
            .map(|&row| prep.design.matrix.get(column * n + row).copied().ok_or_else(out_of_range))
            .collect::<Result<Vec<_>, _>>()?;
        columns.push(values);
    }
    let outcome = rows
        .iter()
        .map(|&row| prep.design.outcome.get(row).copied().ok_or_else(out_of_range))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SubjectColumns {
        nrows: rows.len(),
        columns,
        outcome,
        active: prep.active,
        control: prep.control,
    })
}

impl super::Study {
    /// The independent-subject rows of a prepared panel pulse: the identified adjustment lags and
    /// the treatment of each unit's earliest aligned row.
    pub(crate) fn panel_subject_columns(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<SubjectColumns, CausalError> {
        self.refuse_unlicensed_panel(panel)?;
        if query.is_multi_step_sustained() || !query.effect_modifiers.is_empty() {
            return Err(CausalError::Unsupported {
                message: "independent-subject rows exist for a single-step pulse or one-step \
                          sustained effect without effect modifiers",
            });
        }
        let (identification, estimand, indexer, _) =
            self.identify_pooled_panel(graph, query, ctx)?;
        require_identified(&identification)?;
        let unit_ids = panel_unit_ids(panel);
        let atom = PanelPulseAtom::prepare(
            panel,
            &estimand,
            query,
            &indexer,
            self.split.as_ref(),
            ctx,
            1.0,
        )?;
        let (prep, cluster_ids, panel_times) = atom.stacked(&unit_ids)?;
        subject_columns(&prep, &cluster_ids, &panel_times)
    }
}

/// The contrast design of the subject rows, with the treatment column as the regime.
fn independent_subject_design(
    subjects: &SubjectColumns,
) -> Result<TemporalContrastDesign, CausalError> {
    let x: Vec<f64> = subjects.columns.iter().flatten().copied().collect();
    // Column 1 of a pooled panel design is the treatment.
    let regime = RegimeColumn {
        column: 1,
        active: vec![subjects.active; subjects.nrows],
        control: vec![subjects.control; subjects.nrows],
    };
    TemporalContrastDesign::try_new(
        x,
        subjects.outcome.clone(),
        subjects.nrows,
        vec![regime],
        TrajectoryDependence::IndependentSubjects,
    )
    .map_err(CausalError::from)
}

fn design_note(design: &TemporalContrastDesign) -> &'static str {
    let _ = design;
    "each unit contributes its earliest decision time with a full lag window; its later rows do \
     not enter, so subjects are independent and the construction carries no within-unit \
     dependence correction; the link-scale coefficient on treatment is not the reported contrast"
}

fn link_assumption(description: &'static str) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("temporal.glm_link_contrast.independent_subjects"),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal.glm_link_contrast"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Posterior of an effect from its draws, labelled with the backend that produced them.
pub(super) fn effect_draws_posterior(
    draws: &[f64],
    identification: antecedent_core::IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
    backend: &'static str,
) -> Result<CausalPosterior, CausalError> {
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: Arc::from("ate"),
        }]),
    };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        draws.len(),
        Arc::<[f64]>::from(draws.to_vec()),
    )?;
    Ok(CausalPosterior {
        summaries: draws.summarize(),
        draws,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: InferenceDiagnostics::analytic(backend),
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
    })
}
