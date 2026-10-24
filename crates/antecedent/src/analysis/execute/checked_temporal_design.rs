//! Retained sequentially randomized unit-period designs and regime survival.
//!
//! One table row per unit and period. The arm designs (two arms, several arms, categorical
//! levels, a two-factor factorial, within-cluster interference) and the continuous-dose design
//! carry their recorded assignment law in the query; the excursion target pairs each decision
//! with the same unit's outcome `lag` periods later and is identified by that law
//! ([`antecedent_estimate::temporal_design`]). Frequentist inference reports cluster-robust
//! intervals (whole-procedure cluster bootstrap for the quantile functional; a bias-aware
//! interval and band for the local-quadratic dose grid); Bayesian inference reports the cluster
//! Bayesian-bootstrap posterior of the same functional. Regime survival reads observational
//! trajectories: its frequentist interval refits the treatment and censoring models in every
//! subject-bootstrap replicate, and its posterior is the joint-likelihood parametric g-formula.
//! Every route is unlicensed and its calibration is unmeasured.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_quasi_common::{
    assemble_posterior, effect_quantity, model_assumption, require_built_in_model, scalar_quantity,
};
use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceDiagnostics, InferenceMode, OutcomeExprId, OverlapPolicy, PhysicalExecutionPlan,
    RandomizedTrialOperation, RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, finish_identified_execute_with_context,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery, TemporalDoseTarget,
    TemporalRegimeMethod,
};
use antecedent_estimate::temporal_design::contrasts::{
    DesignScores, factorial_scores, interference_scores, multi_arm_scores, two_arm_scores,
};
use antecedent_estimate::temporal_design::dose::{DosePairs, DoseTarget, fit_dose};
use antecedent_estimate::temporal_design::misclassification::{
    ValidationCounts, correct_posterior, correct_sampling,
};
use antecedent_estimate::temporal_design::observational::{
    OBSERVATIONAL_COORDINATES, ObservationalRows, fit_observational_excursion,
};
use antecedent_estimate::temporal_design::quantile::{QUANTILE_COORDINATES, fit_quantile};
use antecedent_estimate::temporal_design::regime::{
    REGIME_COORDINATES, RegimeMethod, RegimePanel, RegimeRows, StaticRegimes, fit_static_regimes,
};
use antecedent_estimate::temporal_design::survival::{
    SURVIVAL_COORDINATES, Trajectories, TrajectoryRows, fit_regime_survival,
    fit_regime_survival_posterior,
};
use antecedent_estimate::temporal_design::{
    ClusterScores, DEFAULT_BAND_REPLICATES, PanelRows, PosteriorSummary, SamplingSummary,
    bayesian_bootstrap, dense_labels, holm_adjusted, lagged_pairs, sampling_summary, sup_t_band,
};

/// Calibration status of every interval this route reports.
const CALIBRATION: &str = "unmeasured";

/// What a prepared temporal design keeps beside its query: the Bayesian configuration of a
/// posterior study and the whole-procedure bootstrap replicates of a frequentist one.
#[derive(Clone, Debug)]
pub(crate) struct TemporalDesignSettings {
    bayesian: Option<super::BayesianConfig>,
    replicates: u32,
}

/// Frozen temporal design.
pub(crate) type CheckedTemporalDesignOperation =
    RandomizedTrialOperation<TemporalDesignQuery, TemporalDesignSettings>;

impl RandomizedTrialQuery for TemporalDesignQuery {
    const OPERATION: &'static str = "CheckedTemporalDesignOperation";
    const REFRESH_MISMATCH: &'static str =
        "temporal design refresh requires the prepared schema and row order";

    fn describe(&self, rows: usize, debug: &mut std::fmt::DebugStruct<'_, '_>) {
        debug.field("design", &self.design_name());
        debug.field("lag", &self.lag);
        debug.field("rows", &rows);
    }
}

/// Whether the design's frequentist interval is a whole-procedure bootstrap.
fn reads_replicates(query: &TemporalDesignQuery) -> bool {
    matches!(
        query.assignment,
        TemporalAssignment::RegimeSurvival { .. }
            | TemporalAssignment::ObservationalExcursion { .. }
            | TemporalAssignment::ObservationalRegime { .. }
    ) || matches!(query.functional, TemporalDesignFunctional::Quantile { .. })
}

fn observational_design(query: &TemporalDesignQuery) -> bool {
    matches!(query.assignment, TemporalAssignment::ObservationalExcursion { .. })
}

fn regime_design(query: &TemporalDesignQuery) -> bool {
    matches!(query.assignment, TemporalAssignment::ObservationalRegime { .. })
}

fn survival_design(query: &TemporalDesignQuery) -> bool {
    matches!(query.assignment, TemporalAssignment::RegimeSurvival { .. })
}

impl CheckedTemporalDesignOperation {
    /// The design identification this operation records; the route lowering retains it.
    pub(crate) fn design_identification(
        query: &TemporalDesignQuery,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        design_identification(query)
    }

    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::TemporalDesign(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "temporal design operation requires TemporalDesignQuery".into(),
            });
        };
        query.validate()?;
        if query.rows() != data.row_count()
            || study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "a temporal design reads one row per unit and period, with no graph, \
                          refuter or custom validator",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                require_built_in_model(config, "the Bayesian temporal design")?;
                Some(config.clone())
            }
        };
        let replicates = study.bootstrap_replicates;
        if replicates != 0 && (bayesian.is_some() || !reads_replicates(query)) {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "this temporal design does not read bootstrap replicates: its frequentist \
                 interval is analytic and its posterior draws whole units. Next attempt: omit \
                 bootstrap_replicates, or ask for the quantile functional or regime survival, \
                 whose intervals are whole-procedure bootstraps"
            ));
        }
        for id in query.columns() {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let estimator_id = estimator_for(query, bayesian.is_some());
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator_id)
        {
            return Err(CausalError::Compile {
                message: "temporal design query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = design_identification(query);
        Ok(Self::freeze(
            study,
            query,
            data,
            physical,
            identification,
            estimand,
            TemporalDesignSettings { bayesian, replicates },
        ))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let started = std::time::Instant::now();
        let output = self.fit(data, ctx)?;
        let estimator_id = estimator_for(&self.query, self.extra.bayesian.is_some());
        let (payload, posterior) = self.assemble(output)?;
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.identification.clone(),
                self.estimand.clone(),
                EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                Strategy::new(IdentifierId::RandomizedDesign, estimator_id),
                self.query.outcome,
                self.query.outcome,
            )
            .identify_cached(true)
            .wall_time_ns(super::elapsed_ns(started))
            .extra_diagnostics(vec![Diagnostic::new(
                "estimate.temporal_design.unlicensed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "unlicensed and uncalibrated route ({} design, {}): the identifying premises \
                     are declared and not checked, and no repeated-sampling calibration of the \
                     reported uncertainty has been measured",
                    payload.design, payload.uncertainty
                ),
            )])
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws,
                ..Default::default()
            }),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.support_status = None;
        result.payload = crate::result::ResultPayload::TemporalDesign(Box::new(payload));
        Ok(result)
    }

    /// Run the design's procedure.
    #[allow(
        clippy::too_many_lines,
        reason = "a single exhaustive dispatch over the temporal design's assignment and functional"
    )]
    fn fit(&self, data: &TabularData, ctx: &ExecutionContext) -> Result<FitOutput, CausalError> {
        let query = &self.query;
        let outcome = read_column(data, query.outcome)?;
        let (unit, _) = dense_labels(&query.units);
        let draws = self.extra.bayesian.as_ref().map(|config| config.n_draws);
        let floor = query.min_probability;
        if let TemporalAssignment::RegimeSurvival { treatment, censored, confounders, horizon } =
            &query.assignment
        {
            return self.fit_survival(
                data,
                &outcome,
                &unit,
                (*treatment, *censored, &confounders[..], *horizon),
                ctx,
            );
        }
        if let TemporalAssignment::ObservationalRegime {
            treatment,
            confounders,
            active,
            reference,
            method,
        } = &query.assignment
        {
            return self.fit_regime(
                data,
                &outcome,
                &unit,
                (*treatment, &confounders[..], &active[..], &reference[..], *method),
                ctx,
            );
        }
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &query.periods }, query.lag)
            .map_err(CausalError::from)?;
        if let TemporalAssignment::ObservationalExcursion { treatment, history } = &query.assignment
        {
            return self.fit_observational(
                data,
                &outcome,
                &unit,
                &pairs,
                (*treatment, &history[..]),
                ctx,
            );
        }
        if let TemporalDesignFunctional::Quantile { level } = query.functional {
            let TemporalAssignment::TwoArm { assignment, probability } = &query.assignment else {
                return Err(CausalError::Compile {
                    message: "the quantile functional lost its two-arm design".into(),
                });
            };
            let fit = fit_quantile(
                &pairs,
                &outcome,
                assignment,
                probability,
                &unit,
                floor,
                level,
                self.extra.replicates as usize,
                draws,
                ctx,
            )
            .map_err(CausalError::from)?;
            let coordinates = QUANTILE_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
            let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
                (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
                (None, Some(se), Some(interval)) => Summary::Bootstrap {
                    estimate: fit.estimate.to_vec(),
                    standard_error: se.to_vec(),
                    interval: interval.to_vec(),
                },
                _ => {
                    return Err(CausalError::Compile {
                        message: "the quantile fit reported no uncertainty".into(),
                    });
                }
            };
            return Ok(FitOutput {
                coordinates,
                family: vec![0],
                summary,
                pairs: fit.pairs,
                clusters: fit.clusters,
                unpaired: pairs.unpaired,
                minimum_probability: f64::NAN,
                replicates: (fit.posterior.is_none())
                    .then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
                functional: format!("quantile:{level}"),
                uncertainty: if fit.posterior.is_some() {
                    "bayesian_bootstrap_over_units_weighted_quantile"
                } else {
                    "whole_procedure_cluster_bootstrap_percentile"
                },
            });
        }
        if let TemporalAssignment::Dose { dose, density, bandwidth, target } = &query.assignment {
            let input =
                DosePairs { pairs: &pairs, outcome: &outcome, dose, density, cluster: &unit };
            let (estimate_target, label) = dose_target(target);
            let fit =
                fit_dose(input, *bandwidth, &estimate_target, draws, DEFAULT_BAND_REPLICATES, ctx)
                    .map_err(CausalError::from)?;
            let summary = match (fit.sampling, fit.posterior) {
                (Some(sampling), _) => Summary::Sampling(sampling),
                (None, Some(posterior)) => Summary::Posterior(posterior),
                (None, None) => {
                    return Err(CausalError::Compile {
                        message: "the dose fit reported no uncertainty".into(),
                    });
                }
            };
            let uncertainty = match &summary {
                Summary::Sampling(sampling) => sampling.method,
                Summary::Posterior(posterior) => posterior.method,
                Summary::Bootstrap { .. } => "whole_procedure_cluster_bootstrap_percentile",
            };
            let family = if matches!(target, TemporalDoseTarget::SmoothedContrast { .. }) {
                vec![0]
            } else {
                Vec::new()
            };
            let mut output = FitOutput {
                coordinates: fit.coordinates,
                family,
                summary,
                pairs: fit.pairs,
                clusters: fit.clusters,
                unpaired: pairs.unpaired,
                minimum_probability: f64::NAN,
                replicates: None,
                functional: label,
                uncertainty,
            };
            if let (Summary::Posterior(_), Some(bias)) = (&output.summary, fit.smoothing_bias_bound)
            {
                output.functional = format!(
                    "{}; posterior of the bandwidth projection, worst-case smoothing bias {:?} \
                     reported beside it",
                    output.functional, bias
                );
            }
            return Ok(output);
        }
        let scores = self.arm_scores(&pairs, &outcome, &unit, floor)?;
        let width = scores.coordinates.len();
        let accumulated = ClusterScores::accumulate(&scores.scores, width, &scores.cluster_of_pair)
            .map_err(CausalError::from)?;
        let validation = query.outcome_misclassification.map(|counts| ValidationCounts {
            true_positive: counts.true_positive,
            false_negative: counts.false_negative,
            true_negative: counts.true_negative,
            false_positive: counts.false_positive,
        });
        if validation.is_some()
            && outcome
                .iter()
                .any(|y| y.to_bits() != 0.0_f64.to_bits() && y.to_bits() != 1.0_f64.to_bits())
        {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "a misclassification correction reads a binary outcome coded exactly 0 or 1"
            ));
        }
        let mut functional = "mean".to_owned();
        let summary = if let Some(draws) = draws {
            let posterior =
                bayesian_bootstrap(&accumulated, draws, ctx).map_err(CausalError::from)?;
            Summary::Posterior(match validation {
                Some(counts) => {
                    "mean; outcome misclassification corrected".clone_into(&mut functional);
                    correct_posterior(posterior, &scores.coordinates, counts, ctx)
                        .map_err(CausalError::from)?
                }
                None => posterior,
            })
        } else {
            let mut summary = sampling_summary(&accumulated).map_err(CausalError::from)?;
            if let Some(counts) = validation {
                let youden =
                    correct_sampling(&mut summary, &scores.coordinates, &scores.family, counts)
                        .map_err(CausalError::from)?;
                functional =
                    format!("mean; outcome misclassification corrected (Youden index {youden})");
                return Ok(FitOutput {
                    coordinates: scores.coordinates,
                    family: scores.family,
                    uncertainty: summary.method,
                    summary: Summary::Sampling(summary),
                    pairs: scores.pairs,
                    clusters: accumulated.clusters(),
                    unpaired: pairs.unpaired,
                    minimum_probability: scores.minimum_probability,
                    replicates: None,
                    functional,
                });
            }
            if scores.band.len() > 1 {
                sup_t_band(&accumulated, &mut summary, &scores.band, DEFAULT_BAND_REPLICATES, ctx)
                    .map_err(CausalError::from)?;
            }
            if scores.family.len() > 1 {
                summary.holm_adjusted_p = Some(holm_adjusted(&summary, &scores.family));
            }
            Summary::Sampling(summary)
        };
        let uncertainty = match &summary {
            Summary::Sampling(sampling) => sampling.method,
            Summary::Posterior(posterior) => posterior.method,
            Summary::Bootstrap { .. } => "whole_procedure_cluster_bootstrap_percentile",
        };
        Ok(FitOutput {
            coordinates: scores.coordinates,
            family: scores.family,
            summary,
            pairs: scores.pairs,
            clusters: accumulated.clusters(),
            unpaired: pairs.unpaired,
            minimum_probability: scores.minimum_probability,
            replicates: None,
            functional,
            uncertainty,
        })
    }

    /// The pair scores of an arm design.
    fn arm_scores(
        &self,
        pairs: &antecedent_estimate::temporal_design::LaggedPairs,
        outcome: &[f64],
        unit: &[usize],
        floor: f64,
    ) -> Result<DesignScores, CausalError> {
        let query = &self.query;
        let scores = match &query.assignment {
            TemporalAssignment::TwoArm { assignment, probability } => {
                two_arm_scores(pairs, outcome, assignment, probability, unit, floor)
            }
            TemporalAssignment::MultiArm { assignment, probabilities, arms, ordered, .. } => {
                let labels: Vec<String> = arms.iter().map(ToString::to_string).collect();
                multi_arm_scores(
                    pairs,
                    outcome,
                    assignment,
                    probabilities,
                    &labels,
                    unit,
                    *ordered,
                    floor,
                )
            }
            TemporalAssignment::Factorial {
                first,
                first_probability,
                second,
                second_probability,
            } => factorial_scores(
                pairs,
                outcome,
                first,
                first_probability,
                second,
                second_probability,
                unit,
                floor,
            ),
            TemporalAssignment::Interference { assignment, probability, clusters } => {
                let (group, _) = dense_labels(clusters);
                interference_scores(
                    pairs,
                    outcome,
                    assignment,
                    probability,
                    &group,
                    &query.periods,
                    floor,
                )
            }
            TemporalAssignment::Dose { .. }
            | TemporalAssignment::RegimeSurvival { .. }
            | TemporalAssignment::ObservationalExcursion { .. }
            | TemporalAssignment::ObservationalRegime { .. } => {
                return Err(CausalError::Compile {
                    message: "an arm score was requested for a dose, survival or observational \
                              design"
                        .into(),
                });
            }
        };
        scores.map_err(CausalError::from)
    }

    /// The observational excursion: the assignment law is fitted on the measured history, and
    /// the fit is refitted inside every bootstrap replicate or posterior draw.
    fn fit_observational(
        &self,
        data: &TabularData,
        outcome: &[f64],
        unit: &[usize],
        pairs: &antecedent_estimate::temporal_design::LaggedPairs,
        columns: (antecedent_core::VariableId, &[antecedent_core::VariableId]),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, history_ids) = columns;
        let treated = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let history: Vec<Vec<f64>> =
            history_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let rows = ObservationalRows { pairs, outcome, treated: &treated, history: &history, unit };
        let floor = self.query.min_probability;
        let (prior_scale, draws) = self
            .extra
            .bayesian
            .as_ref()
            .map_or((1.0, None), |config| (config.prior_scale, Some(config.n_draws)));
        let fit = fit_observational_excursion(
            rows,
            floor,
            prior_scale,
            self.extra.replicates as usize,
            draws,
            ctx,
        )
        .map_err(CausalError::from)?;
        let coordinates = OBSERVATIONAL_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the observational excursion reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.pairs,
            clusters: fit.clusters,
            unpaired: pairs.unpaired,
            minimum_probability: fit.minimum_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: format!("observational_excursion:history_columns={}", history.len()),
            uncertainty: if posterior {
                "weighted_likelihood_bootstrap_refitting_assignment_model"
            } else {
                "whole_procedure_unit_bootstrap_refitting_assignment_model"
            },
        })
    }

    /// The value of whole static regimes on a continuous end-of-follow-up outcome: every
    /// treatment and outcome model is refitted inside each bootstrap replicate or posterior draw.
    fn fit_regime(
        &self,
        data: &TabularData,
        outcome: &[f64],
        unit: &[usize],
        columns: (
            antecedent_core::VariableId,
            &[antecedent_core::VariableId],
            &[bool],
            &[bool],
            TemporalRegimeMethod,
        ),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, confounder_ids, active, reference, method) = columns;
        let treated = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let confounders: Vec<Vec<f64>> =
            confounder_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let panel = RegimePanel::new(RegimeRows {
            unit,
            period: &self.query.periods,
            outcome,
            treated: &treated,
            confounders: &confounders,
        })
        .map_err(CausalError::from)?;
        let estimator = match method {
            TemporalRegimeMethod::GFormula => RegimeMethod::GFormula,
            TemporalRegimeMethod::InverseProbability => RegimeMethod::InverseProbability,
            TemporalRegimeMethod::DoublyRobust => RegimeMethod::DoublyRobust,
        };
        let (prior_scale, draws) = self
            .extra
            .bayesian
            .as_ref()
            .map_or((1.0, None), |config| (config.prior_scale, Some(config.n_draws)));
        let fit = fit_static_regimes(
            &panel,
            StaticRegimes { active, reference },
            estimator,
            self.query.min_probability,
            prior_scale,
            self.extra.replicates as usize,
            draws,
            ctx,
        )
        .map_err(CausalError::from)?;
        let coordinates = REGIME_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the regime contrast reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.units * fit.periods,
            clusters: fit.units,
            unpaired: 0,
            minimum_probability: fit.minimum_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: format!(
                "observational_regime:method={};periods={};confounder_columns={}",
                method.name(),
                fit.periods,
                confounders.len()
            ),
            uncertainty: if posterior {
                "weighted_likelihood_bootstrap_refitting_regime_models"
            } else {
                "whole_procedure_unit_bootstrap_refitting_regime_models"
            },
        })
    }

    /// Regime survival on subject trajectories.
    fn fit_survival(
        &self,
        data: &TabularData,
        event: &[f64],
        unit: &[usize],
        columns: (
            antecedent_core::VariableId,
            antecedent_core::VariableId,
            &[antecedent_core::VariableId],
            u32,
        ),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, censored_id, confounder_ids, horizon) = columns;
        let treatment = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let censored = indicator(&read_column(data, censored_id)?, "censoring")?;
        let event = indicator(event, "event")?;
        let confounders: Vec<Vec<f64>> =
            confounder_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let rows = TrajectoryRows {
            unit,
            period: &self.query.periods,
            treatment: &treatment,
            censored: &censored,
            event: &event,
            confounders: &confounders,
        };
        let trajectories = Trajectories::new(rows, horizon as usize).map_err(CausalError::from)?;
        let floor = self.query.min_probability;
        let fit = match &self.extra.bayesian {
            Some(config) => fit_regime_survival_posterior(
                &trajectories,
                floor,
                config.prior_scale,
                config.n_draws,
                ctx,
            ),
            None => fit_regime_survival(&trajectories, floor, self.extra.replicates as usize, ctx),
        }
        .map_err(CausalError::from)?;
        let coordinates = SURVIVAL_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the regime survival fit reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.person_periods,
            clusters: fit.subjects,
            unpaired: 0,
            minimum_probability: fit.minimum_regime_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: match fit.posterior_positivity_shortfall {
                Some(shortfall) => format!(
                    "survival:horizon={horizon}; posterior share of draws below the positivity \
                     floor {shortfall}"
                ),
                None => format!("survival:horizon={horizon}"),
            },
            uncertainty: if posterior {
                "joint_likelihood_parametric_g_formula_posterior"
            } else {
                "whole_procedure_subject_bootstrap_refitting_treatment_and_censoring_models"
            },
        })
    }

    /// The result section and, for a Bayesian study, the posterior.
    fn assemble(
        &self,
        output: FitOutput,
    ) -> Result<
        (crate::result::TemporalDesignEstimate, Option<antecedent_estimate::CausalPosterior>),
        CausalError,
    > {
        let k = output.coordinates.len();
        let coordinates: Arc<[Arc<str>]> =
            output.coordinates.iter().map(|name| Arc::<str>::from(name.as_str())).collect();
        let mut section = crate::result::TemporalDesignEstimate {
            design: Arc::from(self.query.design_name()),
            functional: Arc::from(output.functional.as_str()),
            coordinates,
            estimate: Arc::from([]),
            standard_error: Arc::from([]),
            lower: Arc::from([]),
            upper: Arc::from([]),
            covariance: None,
            band_coordinates: Arc::from([]),
            band_critical_value: None,
            band_lower: None,
            band_upper: None,
            holm_adjusted_p: None,
            smoothing_bias_bound: None,
            pairs: output.pairs as u64,
            clusters: output.clusters as u64,
            unpaired: output.unpaired as u64,
            minimum_probability: output.minimum_probability,
            replicates: output.replicates,
            draws: None,
            uncertainty: Arc::from(output.uncertainty),
            calibration: Arc::from(CALIBRATION),
            licensed: false,
        };
        let posterior = match output.summary {
            Summary::Sampling(sampling) => {
                fill_sampling(&mut section, sampling);
                None
            }
            Summary::Bootstrap { estimate, standard_error, interval } => {
                section.estimate = Arc::from(estimate);
                section.standard_error = Arc::from(standard_error);
                section.lower = interval.iter().map(|(low, _)| *low).collect();
                section.upper = interval.iter().map(|(_, high)| *high).collect();
                None
            }
            Summary::Posterior(posterior) => {
                section.estimate = Arc::from(posterior.mean.as_slice());
                section.standard_error = Arc::from(posterior.standard_deviation.as_slice());
                section.lower = Arc::from(posterior.lower.as_slice());
                section.upper = Arc::from(posterior.upper.as_slice());
                section.covariance = Some(draw_covariance(&posterior.draws).into());
                section.draws = posterior.draws.first().map(|column| column.len() as u64);
                Some(self.posterior(&output.coordinates, &output.family, posterior)?)
            }
        };
        if section.estimate.len() != k {
            return Err(CausalError::Compile {
                message: "the temporal design result lost a coordinate".into(),
            });
        }
        Ok((section, posterior))
    }

    fn posterior(
        &self,
        names: &[String],
        family: &[usize],
        posterior: PosteriorSummary,
    ) -> Result<antecedent_estimate::CausalPosterior, CausalError> {
        let method = posterior.method;
        let columns: Vec<_> = posterior
            .draws
            .into_iter()
            .enumerate()
            .map(|(index, draws)| {
                let kind = if index == 0 || family.contains(&index) {
                    effect_quantity(&names[index])
                } else {
                    scalar_quantity(&names[index])
                };
                (kind, draws)
            })
            .collect();
        let mut assumptions = self.identification.required_assumptions.clone();
        assumptions.push(model_assumption(
            method,
            if survival_design(&self.query) {
                "joint_likelihood_factor_posteriors"
            } else if observational_design(&self.query) {
                "weighted_likelihood_bootstrap_assignment_model"
            } else if regime_design(&self.query) {
                "weighted_likelihood_bootstrap_regime_models"
            } else {
                "dirichlet_process_unit_law"
            },
            if observational_design(&self.query) {
                "independent units carry Dirichlet weights; each draw refits the pooled logistic \
                 assignment model under those weights with an isotropic Gaussian coefficient \
                 prior and recomputes the weighted ratio, so uncertainty about the assignment \
                 law is inside the posterior; the prior supplies no identifying premise"
            } else if regime_design(&self.query) {
                "independent units carry Dirichlet weights; each draw refits every per-period \
                 treatment model (isotropic Gaussian coefficient prior) and outcome regression \
                 under those weights and recomputes both regime values, so uncertainty about the \
                 nuisance laws is inside the posterior; the prior supplies no identifying premise"
            } else if survival_design(&self.query) {
                "the confounder transition law is conjugate Gaussian and the hazard, treatment and \
                 censoring laws are pooled logistic with an isotropic Gaussian coefficient prior \
                 (Laplace approximation); the prior supplies no identifying premise"
            } else {
                "independent units have a vanishing-concentration Dirichlet-process prior and each \
                 draw reweights whole units; the recorded assignment law is held fixed, so the \
                 prior supplies no identifying premise"
            },
        ));
        let mut inference = InferenceDiagnostics::analytic(method);
        inference.notes = vec![Arc::from("interval=equal_tailed_95 calibration=unmeasured")];
        assemble_posterior(&columns, &self.identification, assumptions, inference)
    }
}

impl Study {
    pub(super) fn execute_temporal_design(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedTemporalDesignOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// The estimator of a design under an inference mode.
fn estimator_for(query: &TemporalDesignQuery, bayesian: bool) -> EstimatorId {
    match (survival_design(query), bayesian) {
        (false, false) => EstimatorId::TemporalDesignExcursion,
        (false, true) => EstimatorId::BayesianTemporalDesignExcursion,
        (true, false) => EstimatorId::TemporalRegimeSurvival,
        (true, true) => EstimatorId::BayesianTemporalRegimeSurvival,
    }
}

/// One design fit before it becomes a result section.
struct FitOutput {
    coordinates: Vec<String>,
    family: Vec<usize>,
    summary: Summary,
    pairs: usize,
    clusters: usize,
    unpaired: usize,
    minimum_probability: f64,
    replicates: Option<(u64, u64)>,
    functional: String,
    uncertainty: &'static str,
}

enum Summary {
    Sampling(SamplingSummary),
    Bootstrap { estimate: Vec<f64>, standard_error: Vec<f64>, interval: Vec<(f64, f64)> },
    Posterior(PosteriorSummary),
}

fn fill_sampling(section: &mut crate::result::TemporalDesignEstimate, sampling: SamplingSummary) {
    section.estimate = Arc::from(sampling.estimate.as_slice());
    section.standard_error = Arc::from(sampling.standard_error.as_slice());
    section.lower = Arc::from(sampling.lower.as_slice());
    section.upper = Arc::from(sampling.upper.as_slice());
    section.covariance = Some(Arc::from(sampling.covariance.as_slice()));
    if let Some(band) = sampling.band {
        section.band_coordinates = Arc::from(band.coordinates.as_slice());
        section.band_critical_value = Some(band.critical_value);
        section.band_lower = Some(Arc::from(band.lower.as_slice()));
        section.band_upper = Some(Arc::from(band.upper.as_slice()));
    }
    section.holm_adjusted_p = sampling.holm_adjusted_p.map(|values| Arc::from(values.as_slice()));
    section.smoothing_bias_bound =
        sampling.smoothing_bias_bound.map(|values| Arc::from(values.as_slice()));
}

/// Row-major covariance of coordinate-major draws.
fn draw_covariance(draws: &[Vec<f64>]) -> Vec<f64> {
    let k = draws.len();
    let n = draws.first().map_or(0, Vec::len);
    let means: Vec<f64> =
        draws.iter().map(|column| column.iter().sum::<f64>() / n as f64).collect();
    let mut covariance = vec![0.0; k * k];
    if n < 2 {
        return covariance;
    }
    for a in 0..k {
        for b in 0..k {
            let sum: f64 =
                draws[a].iter().zip(&draws[b]).map(|(x, y)| (x - means[a]) * (y - means[b])).sum();
            covariance[a * k + b] = sum / (n as f64 - 1.0);
        }
    }
    covariance
}

/// The estimator's dose target and its label.
fn dose_target(target: &TemporalDoseTarget) -> (DoseTarget, String) {
    match target {
        TemporalDoseTarget::SmoothedCurve { grid } => {
            (DoseTarget::SmoothedCurve { grid: grid.to_vec() }, "dose.smoothed_curve".to_owned())
        }
        TemporalDoseTarget::SmoothedContrast { active, control } => (
            DoseTarget::SmoothedContrast { active: *active, control: *control },
            "dose.smoothed_contrast".to_owned(),
        ),
        TemporalDoseTarget::LocalQuadratic { grid, derivative, curvature_bound } => (
            DoseTarget::LocalQuadratic {
                grid: grid.to_vec(),
                derivative: *derivative,
                curvature_bound: *curvature_bound,
            },
            if *derivative {
                "dose.local_quadratic_derivative".to_owned()
            } else {
                "dose.local_quadratic_level".to_owned()
            },
        ),
    }
}

/// A complete continuous column.
fn read_column(
    data: &TabularData,
    id: antecedent_core::VariableId,
) -> Result<Vec<f64>, CausalError> {
    match data.column(id).map_err(CausalError::from)? {
        antecedent_data::ColumnView::Float64(column) if column.validity.is_all_valid() => {
            Ok(column.values.as_slice().to_vec())
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal design columns must be complete continuous data",
        }),
    }
}

/// A 0/1-coded indicator column.
fn indicator(values: &[f64], name: &'static str) -> Result<Vec<bool>, CausalError> {
    values
        .iter()
        .map(|value| {
            if value.to_bits() == 1.0_f64.to_bits() {
                Ok(true)
            } else if value.to_bits() == 0.0_f64.to_bits() {
                Ok(false)
            } else {
                Err(crate::compile_reason!(
                    "invalid_argument",
                    "the regime survival {name} column must be coded exactly 0 or 1"
                ))
            }
        })
        .collect()
}

fn declared(id: &str, description: &str) -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

fn standard(assumption: Assumption) -> AssumptionRecord {
    AssumptionRecord {
        assumption,
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

/// Declared assumptions of the design.
fn design_assumptions(query: &TemporalDesignQuery) -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(standard(Assumption::Consistency));
    assumptions.push(standard(Assumption::Positivity));
    match &query.assignment {
        TemporalAssignment::Interference { .. } => {
            assumptions.push(declared(
                "no_interference_across_clusters",
                "a unit's outcome depends on its own assignment and on whether another unit of \
                 its interference cluster was treated in the same period, and on nothing \
                 assigned outside the cluster",
            ));
        }
        _ => assumptions.push(standard(Assumption::NoInterference)),
    }
    if query.outcome_misclassification.is_some() {
        assumptions.push(declared(
            "non_differential_outcome_misclassification",
            "the recorded binary outcome is misclassified with the same sensitivity and \
             specificity in every arm, unit and period, learned on an independent validation \
             sample of the same measurement",
        ));
    }
    if let TemporalAssignment::RegimeSurvival { .. } = &query.assignment {
        assumptions.push(declared(
            "sequential_exchangeability_treatment_and_censoring",
            "each period's treatment and censoring are independent of the potential event times \
             given the measured confounder and treatment history (declared, not checked)",
        ));
        assumptions.push(declared(
            "pooled_period_models",
            "the treatment, censoring and hazard laws are pooled logistic in the current \
             confounders, the current or previous treatment and the scaled period",
        ));
    } else if let TemporalAssignment::ObservationalRegime { method, .. } = &query.assignment {
        assumptions.push(declared(
            "sequential_exchangeability_given_measured_history",
            "each period's treatment is independent of the potential end-of-follow-up outcomes \
             given the measured confounder and treatment history, so no unmeasured time-varying \
             confounder influences both a later treatment and the outcome (declared, not checked)",
        ));
        assumptions.push(declared(
            "sequential_positivity_of_regime_actions",
            "every action of each declared regime has probability above the floor given every \
             measured history consistent with the regime's earlier actions (the fitted \
             probabilities are held to the floor)",
        ));
        assumptions.push(declared(
            "complete_follow_up_of_every_unit",
            "every unit is observed in the same periods and its end-of-follow-up outcome is \
             recorded, so there is no censoring or loss to follow-up to model",
        ));
        assumptions.push(declared(
            "working_models_of_the_regime_estimator",
            match method {
                TemporalRegimeMethod::GFormula => {
                    "each period's pseudo-outcome is linear in the confounder and treatment \
                     history with a treatment-by-current-confounder interaction"
                }
                TemporalRegimeMethod::InverseProbability => {
                    "each period's treatment law is logistic in the confounder and treatment \
                     history"
                }
                TemporalRegimeMethod::DoublyRobust => {
                    "either each period's pseudo-outcome is linear in the history with a \
                     treatment-by-current-confounder interaction, or each period's treatment law \
                     is logistic in the history"
                }
            },
        ));
    } else if let TemporalAssignment::ObservationalExcursion { .. } = &query.assignment {
        assumptions.push(declared(
            "sequential_exchangeability_given_measured_history",
            "each decision's assignment is independent of the potential outcomes given the \
             measured decision-time history columns (declared, not checked); no assignment \
             probability is recorded",
        ));
        assumptions.push(declared(
            "pooled_logistic_assignment_model",
            "the assignment law is pooled logistic in the measured history columns and is refitted \
             inside every bootstrap replicate or posterior draw",
        ));
        assumptions.push(declared(
            "administrative_outcome_follow_up",
            "a decision whose outcome period is not observed lies past the end of follow-up, so \
             dropping it does not depend on the outcome",
        ));
    } else {
        assumptions.push(declared(
            "sequential_randomization_recorded",
            "each decision's assignment was drawn from the recorded probability (or dose density) \
             given the unit's history; the record is the design, not an estimate",
        ));
        assumptions.push(declared(
            "administrative_outcome_follow_up",
            "a decision whose outcome period is not observed lies past the end of follow-up, so \
             dropping it does not depend on the outcome",
        ));
    }
    assumptions
}

fn design_identification(
    query: &TemporalDesignQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = design_assumptions(query);
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution =
        arena.intern_distribution(outcomes, empty, intervention, DomainRef::Observational);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let (rule, note) = if regime_design(query) {
        (
            "temporal_design.observational_regime",
            "sequential exchangeability given the measured confounder and treatment history, with \
             sequential positivity, identifies the mean end-of-follow-up outcome under each \
             static treatment regime",
        )
    } else if observational_design(query) {
        (
            "temporal_design.observational_excursion",
            "sequential exchangeability given the measured decision-time history identifies the \
             excursion contrast of the decision on the outcome lag periods later",
        )
    } else if survival_design(query) {
        (
            "temporal_design.regime_survival",
            "sequential exchangeability of treatment and censoring given the measured history \
             identifies survival under each static regime",
        )
    } else {
        (
            "temporal_design.sequential_randomization",
            "the recorded sequential randomization identifies the excursion contrast of the \
             decision on the outcome lag periods later",
        )
    };
    arena.set_derivation(functional, DerivationMeta::rule(rule, Some(Arc::from(note))));
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, note);
    let identification = IdentificationResult::identified(
        CausalQuery::TemporalDesign(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
