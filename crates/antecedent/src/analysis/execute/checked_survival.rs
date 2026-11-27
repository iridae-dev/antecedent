//! Retained graphless randomized survival execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteFinish, IdentifierId, InferenceMode, OutcomeExprId,
    OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation, RandomizedTrialQuery,
    RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData, VariableId,
    finish_identified_execute_with_context,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    SurvivalFunctional,
};
use antecedent_estimate::survival::{
    KnownCensoring, RandomizedSurvivalSummary, SurvivalBootstrapIntervals, SurvivalEndpoint,
    SurvivalRows, randomized_survival_bootstrap_difference_band_with_observation,
    randomized_survival_ipcw_summary, randomized_survival_summary,
};
use antecedent_stats::survival::{
    ConditionalObservationRows, STRATIFIED_CENSORING_BAND, STRATIFIED_CENSORING_PROVENANCE,
    randomized_survival_bootstrap_intervals_controlled, standardized_conditional_survival,
    survival_event_code,
};

mod bayesian;

/// The extra settings fixed at prepare time: the bootstrap replicate count of the frequentist
/// route and, for the Bayesian route, the study's Bayesian configuration.
pub(crate) type CheckedSurvivalOperation =
    RandomizedTrialOperation<antecedent_core::SurvivalQuery, (u32, Option<super::BayesianConfig>)>;

impl RandomizedTrialQuery for antecedent_core::SurvivalQuery {
    const OPERATION: &'static str = "CheckedSurvivalOperation";
    const REFRESH_MISMATCH: &'static str =
        "survival refresh requires the prepared schema and row order";
}

impl CheckedSurvivalOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Survival(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "survival operation requires SurvivalQuery".into(),
            });
        };
        query.validate()?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.settings.refute != RefuteSuite::None
            || !study.declared.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.declared.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "survival requires a graphless randomized trial and an explicit observation assumption, without a graph, refuter or custom validator",
            });
        }
        let bayesian = match &study.settings.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                if query.stratified_censoring.is_some() {
                    return Err(conditional_refusal(
                        antecedent_core::reason_details::survival::CONDITIONAL_OBSERVATION_INFERENCE,
                        "the fitted conditional survival route requires frequentist inference",
                    ));
                }
                bayesian::require_scope(query, config, study.settings.bootstrap_replicates)?;
                Some(config.clone())
            }
        };
        // The pointwise interval resamples subjects within arm, one row each; a declared clustered
        // panel or dependent series would be read as independent subjects.
        if bayesian.is_none() && study.settings.bootstrap_replicates > 0 {
            super::gcm_bootstrap::require_subject_resampling(study.declared.sampling_design)?;
        }
        for id in
            [Some(query.duration), Some(query.event), Some(query.treatment), query.delayed_entry]
                .into_iter()
                .flatten()
        {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        if let Some(known) = &query.known_censoring {
            for id in known.columns.iter().copied() {
                data.schema().get(id).map_err(CausalError::rejected)?;
            }
        }
        if let Some(learner) = &query.stratified_censoring {
            for id in [learner.entry_stratum, learner.censoring_stratum] {
                data.schema().get(id).map_err(|_| conditional_refusal(
                    antecedent_core::reason_details::survival::CONDITIONAL_OBSERVATION_IDENTIFICATION_INPUTS,
                    "conditional survival requires both declared baseline columns in the subject table",
                ))?;
            }
        }
        let estimator = if bayesian.is_some() {
            EstimatorId::BayesianPiecewiseSurvival
        } else {
            EstimatorId::RandomizedSurvivalProductLimit
        };
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator)
        {
            return Err(CausalError::Compile {
                message: "survival query differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) =
            study.design_identification_or(physical, || survival_identification(query));
        Ok(Self::freeze(
            study,
            query,
            data,
            physical,
            identification,
            estimand,
            (study.settings.bootstrap_replicates, bayesian),
        ))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        if let Some(config) = &self.extra.1 {
            return self.execute_bayesian(data, config, ctx);
        }
        let columns = self.survival_columns(data)?;
        let treated_count = columns.treated.iter().filter(|&&arm| arm).count();
        let assignment_counts = [columns.treated.len() - treated_count, treated_count];
        let endpoint = match self.query.functional {
            SurvivalFunctional::SurvivalAndRmst => SurvivalEndpoint::Survival,
            SurvivalFunctional::CumulativeIncidence { target_cause } => {
                SurvivalEndpoint::CumulativeIncidence { target_cause }
            }
        };
        let rows = SurvivalRows {
            duration: &columns.duration,
            event_code: &columns.event,
            treated: &columns.treated,
            delayed_entry: columns.entry.as_deref(),
        };
        let conditional = self.query.stratified_censoring.is_some();
        let SurvivalEstimateParts {
            summary,
            minimum_censoring_survival,
            intervals,
            difference_band,
            band_unavailable_reason,
        } = if let Some(learner) = &self.query.stratified_censoring {
            self.conditional_estimate(learner, rows, data, endpoint, assignment_counts, ctx)?
        } else {
            self.unconditional_estimate(rows, data, endpoint, assignment_counts, ctx)?
        };
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
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
                Strategy::new(
                    IdentifierId::RandomizedDesign,
                    EstimatorId::RandomizedSurvivalProductLimit,
                ),
                self.query.treatment,
                self.query.duration,
            )
            .identify_cached(true)
            .extra_diagnostics(survival_diagnostics(
                conditional,
                intervals.is_some(),
                minimum_censoring_survival,
                endpoint,
            ))
            .bootstrap_replicates_ok(intervals.as_ref().map(|value| value.replicates_ok)),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        let minimum_horizon_risk_set =
            if !conditional && matches!(endpoint, SurvivalEndpoint::CumulativeIncidence { .. }) {
                antecedent_stats::survival::delayed_entry_horizon_risk_set(&rows, self.query.tau)
            } else {
                None
            };
        result.payload = crate::result::ResultPayload::Survival(self.survival_payload(
            summary,
            minimum_horizon_risk_set,
            intervals.as_ref(),
            minimum_censoring_survival,
            assignment_counts,
            (difference_band, band_unavailable_reason),
        ));
        self.support_survival_intervals(&mut result, intervals.as_ref());
        result.rebind_interval(false);
        Ok(result)
    }

    /// Fits the stratified censoring learner and standardizes the arm curves to the declared population.
    fn conditional_estimate(
        &self,
        learner: &antecedent_core::StratifiedCensoringSurvival,
        rows: SurvivalRows<'_>,
        data: &TabularData,
        endpoint: SurvivalEndpoint,
        assignment_counts: [usize; 2],
        ctx: &ExecutionContext,
    ) -> Result<SurvivalEstimateParts, CausalError> {
        let bytes = survival_work_bytes(self.rows as u64, 240, self.extra.0, 8)?;
        let _scratch = survival_scratch(ctx, bytes)?;
        let fitted = standardized_conditional_survival(
            &ConditionalObservationRows {
                survival: rows,
                entry_stratum: &numeric_column(data, learner.entry_stratum)?,
                censoring_stratum: &numeric_column(data, learner.censoring_stratum)?,
                population_weights: learner.population_weights,
                censoring_floor: learner.minimum_probability,
            },
            self.query.tau,
            endpoint,
            self.extra.0,
            ctx.rng.master_seed(),
            || ctx.cancellation.is_cancelled(),
        )
        .map_err(|message| {
            conditional_error(
                ctx,
                antecedent_core::reason_details::survival::CONDITIONAL_OBSERVATION_SUPPORT,
                message,
            )
        })?;
        let key = antecedent_io::graphless_support::conditional_survival_band_key(
            rows.delayed_entry.is_some(),
            endpoint != SurvivalEndpoint::Survival,
            self.query.tau,
            learner.population_weights,
            learner.minimum_probability,
        );
        Ok(SurvivalEstimateParts {
            summary: fitted.summary,
            minimum_censoring_survival: Some(fitted.minimum_censoring_survival),
            intervals: None,
            difference_band: fitted.band.map(|band| {
                let support = self.band_support(key, assignment_counts, false, band.replicates_ok);
                result_band(band, support)
            }),
            band_unavailable_reason: None,
        })
    }

    /// Reads the product-limit or IPCW point curves, then any bootstrap intervals and difference band.
    fn unconditional_estimate(
        &self,
        rows: SurvivalRows<'_>,
        data: &TabularData,
        endpoint: SurvivalEndpoint,
        assignment_counts: [usize; 2],
        ctx: &ExecutionContext,
    ) -> Result<SurvivalEstimateParts, CausalError> {
        let _scratch = if self.extra.0 > 0 {
            let grid_width =
                self.query.known_censoring.as_ref().map_or(0, |known| known.times.len());
            let stride = 240_u64
                .checked_add(
                    u64::try_from(grid_width)
                        .ok()
                        .and_then(|width| width.checked_mul(24))
                        .ok_or_else(survival_quote_overflow)?,
                )
                .ok_or_else(survival_quote_overflow)?;
            let bytes = survival_work_bytes(self.rows as u64, stride, self.extra.0, 16)?;
            Some(survival_scratch(ctx, bytes)?)
        } else {
            None
        };
        let (summary, minimum_censoring_survival, censoring_grid) =
            self.point_summary(data, &rows, endpoint)?;
        let intervals = self.bootstrap_intervals(&rows, censoring_grid.as_ref(), endpoint, ctx)?;
        let (band, reason) =
            self.difference_band(&rows, censoring_grid.as_ref(), endpoint, assignment_counts, ctx);
        Ok(SurvivalEstimateParts {
            summary,
            minimum_censoring_survival,
            intervals,
            difference_band: band,
            band_unavailable_reason: reason,
        })
    }

    /// Reads the survival columns and validates event codes and the binary arm before any estimator runs.
    fn survival_columns(&self, data: &TabularData) -> Result<SurvivalColumns, CausalError> {
        let duration = numeric_column(data, self.query.duration)?;
        let event = event_codes(data, self.query.event)?;
        let treatment_raw = numeric_column(data, self.query.treatment)?;
        let entry = self.query.delayed_entry.map(|id| numeric_column(data, id)).transpose()?;
        let treated = treatment_raw
            .iter()
            .map(|&v| match v {
                0.0 => Ok(false),
                1.0 => Ok(true),
                _ => Err(CausalError::Unsupported {
                    message: "randomized survival treatment must be binary zero or one",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SurvivalColumns { duration, event, treated, entry })
    }

    /// Point-only or known-censoring IPCW product-limit summary, keeping the supplied censoring grid for the bootstrap.
    fn point_summary<'a>(
        &'a self,
        data: &TabularData,
        rows: &SurvivalRows<'_>,
        endpoint: SurvivalEndpoint,
    ) -> Result<(RandomizedSurvivalSummary, Option<f64>, Option<CensoringGrid<'a>>), CausalError>
    {
        let mut censoring_grid = None;
        let (summary, minimum_censoring_survival) = if let Some(known) = &self.query.known_censoring
        {
            let columns = known
                .columns
                .iter()
                .map(|id| numeric_column(data, *id))
                .collect::<Result<Vec<_>, _>>()?;
            let mut probabilities = Vec::with_capacity(self.rows * columns.len());
            for row in 0..self.rows {
                probabilities.extend(columns.iter().map(|column| column[row]));
            }
            censoring_grid = Some((known.times.as_ref(), probabilities, known.minimum_probability));
            let (_, probabilities, _) = censoring_grid.as_ref().expect("just set");
            let (summary, minimum) = randomized_survival_ipcw_summary(
                rows,
                &KnownCensoring {
                    times: &known.times,
                    survival: probabilities,
                    minimum_probability: known.minimum_probability,
                },
                self.query.tau,
                endpoint,
            )
            .map_err(|message| CausalError::Unsupported { message })?;
            (summary, Some(minimum))
        } else {
            (
                randomized_survival_summary(rows, self.query.tau, endpoint)
                    .map_err(|message| CausalError::Unsupported { message })?,
                None,
            )
        };
        Ok((summary, minimum_censoring_survival, censoring_grid))
    }

    /// Arm-stratified subject bootstrap intervals when replicates were requested.
    fn bootstrap_intervals(
        &self,
        rows: &SurvivalRows<'_>,
        censoring_grid: Option<&CensoringGrid<'_>>,
        endpoint: SurvivalEndpoint,
        ctx: &ExecutionContext,
    ) -> Result<Option<SurvivalBootstrapIntervals>, CausalError> {
        let bootstrap_replicates = self.extra.0;
        Ok(if bootstrap_replicates > 0 {
            Some(
                randomized_survival_bootstrap_intervals_controlled(
                    rows,
                    censoring_grid
                        .map(|(times, probabilities, floor)| KnownCensoring {
                            times,
                            survival: probabilities,
                            minimum_probability: *floor,
                        })
                        .as_ref(),
                    self.query.tau,
                    endpoint,
                    bootstrap_replicates,
                    ctx.rng.master_seed(),
                    || ctx.cancellation.is_cancelled(),
                )
                .map_err(|message| {
                    if ctx.cancellation.is_cancelled() {
                        CausalError::Cancelled {
                            stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                        }
                    } else {
                        CausalError::Unsupported { message }
                    }
                })?,
            )
        } else {
            None
        })
    }

    /// Simultaneous bootstrap difference band, or the reason this design cannot publish one.
    fn difference_band(
        &self,
        rows: &SurvivalRows<'_>,
        censoring_grid: Option<&CensoringGrid<'_>>,
        endpoint: SurvivalEndpoint,
        assignment_counts: [usize; 2],
        ctx: &ExecutionContext,
    ) -> (Option<crate::result::SurvivalDifferenceBand>, Option<Arc<str>>) {
        let bootstrap_replicates = self.extra.0;
        if bootstrap_replicates == 0 {
            (None, None)
        } else if rows.delayed_entry.is_some()
            && bootstrap_replicates < antecedent_stats::survival::SURVIVAL_BAND_MIN_REPLICATES
        {
            (
                None,
                Some(Arc::from(
                    "simultaneous survival band with delayed entry requires 399 valid draws",
                )),
            )
        } else if censoring_grid.is_some() && rows.delayed_entry.is_none() {
            (
                None,
                Some(Arc::from(
                    "simultaneous survival band does not cover caller-supplied censoring weights",
                )),
            )
        } else {
            let known = censoring_grid.map(|(times, probabilities, floor)| KnownCensoring {
                times,
                survival: probabilities,
                minimum_probability: *floor,
            });
            match randomized_survival_bootstrap_difference_band_with_observation(
                rows,
                known.as_ref(),
                self.query.tau,
                endpoint,
                bootstrap_replicates,
                ctx.rng.master_seed() ^ 0x5A7A_BA4D,
                || ctx.cancellation.is_cancelled(),
            ) {
                Ok(band) => {
                    let key = antecedent_io::graphless_support::survival_band_key(
                        rows.delayed_entry.is_some(),
                        known.is_some(),
                        matches!(endpoint, SurvivalEndpoint::CumulativeIncidence { .. }),
                    );
                    let band_support = self.band_support(
                        key,
                        assignment_counts,
                        known.is_some(),
                        band.replicates_ok,
                    );
                    (Some(result_band(band, band_support)), None)
                }
                Err(reason) => (None, Some(Arc::from(reason))),
            }
        }
    }

    fn band_support(
        &self,
        key: Option<antecedent_io::graphless_support::GraphlessSupportKey<'_>>,
        [control, treated]: [usize; 2],
        fixed_known: bool,
        replicates: u32,
    ) -> Option<crate::support::CellStatus> {
        use antecedent_io::graphless_support::GraphlessResampleSupport;
        key.is_some_and(|key| {
            crate::support::admits_resampled(
                key,
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    treated,
                    control,
                    rows: self.rows,
                    interval_95_published: true,
                    reported_intervals: 1,
                    all_reported_intervals: true,
                    known_density: fixed_known,
                    ..Default::default()
                },
                GraphlessResampleSupport {
                    interval_resamples: replicates as usize,
                    ..Default::default()
                },
            )
        })
        .then_some(crate::support::CellStatus::Supported)
    }

    /// Assembles the retained survival estimate from the point summary, bootstrap intervals, and band.
    fn survival_payload(
        &self,
        summary: RandomizedSurvivalSummary,
        minimum_horizon_risk_set: Option<usize>,
        intervals: Option<&SurvivalBootstrapIntervals>,
        minimum_censoring_survival: Option<f64>,
        assignment_counts: [usize; 2],
        band: (Option<crate::result::SurvivalDifferenceBand>, Option<Arc<str>>),
    ) -> crate::SurvivalEstimate {
        let (difference_band, band_unavailable_reason) = band;
        let conditional = self.query.stratified_censoring.is_some();
        let conditional_draws = conditional
            .then_some(difference_band.as_ref())
            .flatten()
            .map(|band| band.replicates_ok);
        crate::SurvivalEstimate {
            has_delayed_entry: self.query.delayed_entry.is_some(),
            times: summary.times.into(),
            control: summary.control.into(),
            treated: summary.treated.into(),
            rmst_control: summary.rmst_control,
            rmst_treated: summary.rmst_treated,
            target_cause: match self.query.functional {
                SurvivalFunctional::SurvivalAndRmst => None,
                SurvivalFunctional::CumulativeIncidence { target_cause } => Some(target_cause),
            },
            tau: self.query.tau,
            minimum_event_risk_set: summary.minimum_event_risk_set,
            minimum_horizon_risk_set,
            uncertainty: Arc::from(if conditional_draws.is_some() {
                STRATIFIED_CENSORING_BAND
            } else if intervals.is_some() {
                "subject_stratified_percentile_bootstrap_pointwise_95"
            } else {
                "point_only_no_interval"
            }),
            rmst_difference_interval: intervals.and_then(|value| value.rmst_difference),
            difference_at_tau_interval: intervals.map(|value| value.difference_at_tau),
            bootstrap_replicates_requested: intervals
                .as_ref()
                .map(|value| value.replicates_requested)
                .or(conditional_draws.map(|_| self.extra.0)),
            bootstrap_replicates_ok: intervals
                .map(|value| value.replicates_ok)
                .or(conditional_draws),
            assignment_counts,
            censoring_survival_provenance: minimum_censoring_survival.map(|_| {
                Arc::from(if conditional {
                    STRATIFIED_CENSORING_PROVENANCE
                } else {
                    "caller_supplied_fixed_not_fitted_or_verified"
                })
            }),
            difference_band,
            band_unavailable_reason,
        }
    }

    /// Grants the exact pointwise claim when the reported interval(s) clear the support gate.
    fn support_survival_intervals(
        &self,
        result: &mut StudyResult,
        intervals: Option<&SurvivalBootstrapIntervals>,
    ) {
        use antecedent_io::graphless_support::{
            GraphlessResampleSupport, delayed_entry_incidence_support_profile,
            survival_pointwise_key,
        };
        let (Some(intervals), Some(curve)) = (intervals, result.survival()) else { return };
        let incidence = curve.target_cause.is_some();
        let ipcw = self.query.known_censoring.is_some();
        let Some(key) = survival_pointwise_key(self.query.delayed_entry.is_some(), ipcw, incidence)
        else {
            return;
        };
        if intervals.replicates_ok < 299
            || (incidence
                && (curve.difference_band.is_some()
                    || !delayed_entry_incidence_support_profile(
                        curve.minimum_event_risk_set,
                        curve.minimum_horizon_risk_set,
                        curve.bootstrap_replicates_requested,
                        curve.bootstrap_replicates_ok,
                    )))
        {
            return;
        }
        if crate::support::admits_resampled(
            key,
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "unit",
                treated: curve.assignment_counts[1],
                control: curve.assignment_counts[0],
                rows: self.rows,
                interval_95_published: true,
                reported_intervals: 1 + usize::from(intervals.rmst_difference.is_some()),
                all_reported_intervals: intervals.rmst_difference.is_some() != incidence,
                known_density: ipcw,
                ..Default::default()
            },
            GraphlessResampleSupport {
                interval_resamples: intervals.replicates_ok as usize,
                ..Default::default()
            },
        ) {
            result.stamp_supported(crate::support::graphless_unit(key));
        }
    }
}

/// The estimator, censoring-support and route diagnostics of one survival execution.
fn survival_diagnostics(
    conditional: bool,
    bootstrapped: bool,
    minimum_censoring_survival: Option<f64>,
    endpoint: SurvivalEndpoint,
) -> Vec<Diagnostic> {
    vec![Diagnostic::new(
                    if conditional { "estimate.survival.standardized_conditional" } else if bootstrapped { "estimate.survival.pointwise_bootstrap" } else { "estimate.survival.point_only" },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if conditional {
                        "standardizes the arm curves to the declared pre-entry population using stratified fitted censoring survival; each requested band draw refits the censoring learner on resampled subjects"
                    } else if bootstrapped && minimum_censoring_survival.is_some() {
                        "arm-stratified subject bootstrap with caller-supplied censoring survival held fixed; scalar intervals are pointwise and do not include nuisance-model uncertainty"
                    } else if bootstrapped && matches!(endpoint, SurvivalEndpoint::CumulativeIncidence { .. }) {
                        "arm-stratified subject bootstrap for the target-cause cumulative-incidence difference at the declared horizon"
                    } else if bootstrapped {
                        "arm-stratified subject bootstrap for RMST and restriction-horizon contrasts; scalar intervals are pointwise, not curve bands"
                    } else if minimum_censoring_survival.is_some() {
                        "randomized IPCW estimate using caller-supplied censoring survival; no interval is reported"
                    } else {
                        "randomized product-limit risk-set estimate; independent censoring and entry are declared; no interval or calibrated uncertainty is reported"
                    },
                )].into_iter().chain(minimum_censoring_survival.map(|minimum| Diagnostic::new(
                    "estimate.survival.censoring_support",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!("minimum {ownership} censoring survival: {minimum:.6}", ownership = if conditional { "fitted" } else { "supplied" }),
                ))).chain(minimum_censoring_survival.map(|_| {
                    // Every row's supplied censoring survival cleared the positivity floor on the
                    // whole grid (a row below it is a refusal), so no mass was left unevaluated.
                    antecedent_core::diagnostic::RouteNativeDiagnostic::check(
                        antecedent_core::diagnostic::RouteKind::Censoring,
                        antecedent_core::diagnostic::RouteCheck::new(
                            false,
                            antecedent_core::diagnostic::RouteMass::zero(),
                        ),
                    )
                    .to_diagnostic()
                })).collect()
}

/// Point curves and any intervals or difference band one survival estimator reported.
struct SurvivalEstimateParts {
    summary: RandomizedSurvivalSummary,
    minimum_censoring_survival: Option<f64>,
    intervals: Option<SurvivalBootstrapIntervals>,
    difference_band: Option<crate::result::SurvivalDifferenceBand>,
    band_unavailable_reason: Option<Arc<str>>,
}

fn result_band(
    band: antecedent_stats::survival::SurvivalDifferenceBand,
    support_status: Option<crate::support::CellStatus>,
) -> crate::result::SurvivalDifferenceBand {
    crate::result::SurvivalDifferenceBand {
        times: band.times.into(),
        difference: band.difference.into(),
        lower: band.lower.into(),
        upper: band.upper.into(),
        replicates_ok: band.replicates_ok,
        minimum_horizon_risk_set: band.minimum_horizon_risk_set,
        support_status,
    }
}

fn survival_quote_overflow() -> CausalError {
    CausalError::Resource { message: "survival work quote overflowed".into() }
}

fn survival_work_bytes(
    rows: u64,
    row_stride: u64,
    replicates: u32,
    per_replicate: u64,
) -> Result<u64, CausalError> {
    let body = rows.checked_mul(row_stride).ok_or_else(survival_quote_overflow)?;
    let draws =
        u64::from(replicates).checked_mul(per_replicate).ok_or_else(survival_quote_overflow)?;
    body.checked_add(draws).ok_or_else(survival_quote_overflow)
}

fn survival_scratch(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

fn conditional_refusal(detail: &str, message: &str) -> CausalError {
    CausalError::Estimate(antecedent_estimate::EstimationError::refused_detail(
        antecedent_core::reason_code!("route_not_supported"),
        detail,
        message,
    ))
}

fn conditional_error(ctx: &ExecutionContext, detail: &str, message: &str) -> CausalError {
    if ctx.cancellation.is_cancelled() {
        CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT }
    } else {
        conditional_refusal(detail, message)
    }
}

/// Validated survival columns the estimators and bootstrap read.
struct SurvivalColumns {
    duration: Vec<f64>,
    event: Vec<i64>,
    treated: Vec<bool>,
    entry: Option<Vec<f64>>,
}

/// Caller-supplied censoring survival: grid times, row-major probabilities, and the probability floor.
type CensoringGrid<'a> = (&'a [f64], Vec<f64>, f64);

fn event_codes(data: &TabularData, id: VariableId) -> Result<Vec<i64>, CausalError> {
    use antecedent_data::ColumnView;
    let invalid_code = || CausalError::Unsupported {
        message: "survival event codes must be finite nonnegative integers within the signed 64-bit range",
    };
    match data.column(id).map_err(CausalError::from)? {
        ColumnView::Float64(col) if col.validity.is_all_valid() => col
            .values
            .iter()
            .map(|&value| survival_event_code(value).ok_or_else(invalid_code))
            .collect(),
        ColumnView::Int64(col) if col.validity.is_all_valid() => {
            if col.values.iter().any(|&value| value < 0) {
                return Err(invalid_code());
            }
            Ok(col.values.to_vec())
        }
        ColumnView::Boolean(col) if col.validity.is_all_valid() => {
            Ok(col.values.iter().map(|&value| i64::from(value)).collect())
        }
        _ => Err(CausalError::Unsupported {
            message: "survival duration, event, treatment, and entry columns must be complete numeric data",
        }),
    }
}

fn numeric_column(data: &TabularData, id: VariableId) -> Result<Vec<f64>, CausalError> {
    match data.column(id).map_err(CausalError::from)? {
        antecedent_data::ColumnView::Float64(col) if col.validity.is_all_valid() => {
            Ok(col.values.to_vec())
        }
        antecedent_data::ColumnView::Int64(col) if col.validity.is_all_valid() => {
            Ok(col.values.iter().map(|&v| v as f64).collect())
        }
        antecedent_data::ColumnView::Boolean(col) if col.validity.is_all_valid() => {
            Ok(col.values.iter().map(|&v| f64::from(v)).collect())
        }
        _ => Err(CausalError::Unsupported {
            message: "survival duration, event, treatment, and entry columns must be complete numeric data",
        }),
    }
}

impl Study {
    pub(super) fn execute_survival(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedSurvivalOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn survival_identification(
    query: &antecedent_core::SurvivalQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for assumption in [Assumption::Consistency, Assumption::NoInterference, Assumption::Positivity]
    {
        assumptions.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    for (id, description) in [
        (
            "known_random_assignment",
            "treatment is randomized independently of potential event times",
        ),
        (
            "independent_censoring_and_entry",
            if query.stratified_censoring.is_some() {
                "given arm and entry stratum, entry is independent of event, cause, censoring and the censoring stratum; given arm and both baseline strata, censoring is independent of event and cause"
            } else if query.known_censoring.is_some() && query.delayed_entry.is_some() {
                "left entry and censoring are marginally independent of potential event times within treatment arms, and the caller-supplied censoring survival is correct"
            } else if query.known_censoring.is_some() {
                "censoring is independent of potential event times within treatment arms conditional on declared variables, and the caller-supplied censoring survival is correct"
            } else {
                "censoring and delayed entry are marginally independent of potential event times within treatment arms"
            },
        ),
        (
            "common_restriction_horizon",
            "the declared restriction horizon is observed in both treatment arms",
        ),
    ] {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.known_censoring.is_some() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("known_censoring_survival_positivity"),
                description: Arc::from("the supplied subject-specific censoring survival is outcome-independent, row-aligned, and positive through the restriction horizon"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.stratified_censoring.is_some() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("pre_entry_standardization_population"),
                description: Arc::from("the declared probabilities describe the two entry strata in the randomized population before truncation; each arm and baseline history has positive risk and censoring survival through the horizon"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.duration]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution =
        arena.intern_distribution(outcomes, empty, intervention, DomainRef::Observational);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.duration),
        distribution,
    });
    let derivation_text = if query.stratified_censoring.is_some() {
        "weight entry-stratum risk sets by inverse fitted censoring left limits, then mix the reconstructed arm distributions using the fixed pre-entry population probabilities"
    } else {
        "arm survival or cause-specific cumulative incidence is identified under random assignment and independent censoring/entry"
    };
    arena.set_derivation(functional, DerivationMeta::rule("randomized.survival_product_limit", Some(Arc::from(if query.stratified_censoring.is_some() { derivation_text } else { "randomized assignment and independent observation identify arm event-time functionals" }))));
    let estimand = IdentifiedEstimand::new(
        "randomized.survival_product_limit",
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push("randomized.survival_product_limit", derivation_text);
    let identification = IdentificationResult::identified(
        CausalQuery::Survival(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
