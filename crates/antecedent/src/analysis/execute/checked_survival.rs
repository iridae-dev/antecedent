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
    SurvivalRows, randomized_survival_bootstrap_difference_band,
    randomized_survival_bootstrap_intervals, randomized_survival_ipcw_summary,
    randomized_survival_summary,
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
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "survival requires a graphless randomized trial and an explicit observation assumption, without a graph, refuter or custom validator",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                bayesian::require_scope(query, config, study.bootstrap_replicates)?;
                Some(config.clone())
            }
        };
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
            (study.bootstrap_replicates, bayesian),
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
        let (summary, minimum_censoring_survival, censoring_grid) =
            self.point_summary(data, &rows, endpoint)?;
        let intervals = self.bootstrap_intervals(&rows, censoring_grid.as_ref(), endpoint, ctx)?;
        let (difference_band, band_unavailable_reason) = self.difference_band(
            &columns,
            censoring_grid.is_some(),
            endpoint,
            assignment_counts,
            ctx,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ), Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedSurvivalProductLimit), self.query.treatment, self.query.duration).identify_cached(true).extra_diagnostics(vec![Diagnostic::new(
                    if intervals.is_some() { "estimate.survival.pointwise_bootstrap" } else { "estimate.survival.point_only" },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if intervals.is_some() && minimum_censoring_survival.is_some() {
                        "arm-stratified subject bootstrap with caller-supplied censoring survival held fixed; scalar intervals are pointwise and do not include nuisance-model uncertainty"
                    } else if intervals.is_some() {
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
                    format!("minimum supplied censoring survival: {minimum:.6}"),
                ))).collect()).bootstrap_replicates_ok(intervals.as_ref().map(|value| value.replicates_ok)),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.payload = crate::result::ResultPayload::Survival(self.survival_payload(
            summary,
            intervals.as_ref(),
            minimum_censoring_survival,
            assignment_counts,
            (difference_band, band_unavailable_reason),
        ));
        self.license_survival_intervals(&mut result, intervals.as_ref(), assignment_counts);
        Ok(result)
    }

    /// Reads the survival columns and validates event codes and the binary arm before any estimator runs.
    fn survival_columns(&self, data: &TabularData) -> Result<SurvivalColumns, CausalError> {
        let duration = numeric_column(data, self.query.duration)?;
        let event_raw = numeric_column(data, self.query.event)?;
        let treatment_raw = numeric_column(data, self.query.treatment)?;
        let entry = self.query.delayed_entry.map(|id| numeric_column(data, id)).transpose()?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "each value is validated finite, integral, nonnegative, and within i64 range on the line above the cast"
        )]
        let event = event_raw
            .iter()
            .map(|&v| {
                if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > i64::MAX as f64 {
                    Err(CausalError::Unsupported {
                        message: "survival event codes must be finite nonnegative integers",
                    })
                } else {
                    Ok(v as i64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
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
                randomized_survival_bootstrap_intervals(
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
                )
                .map_err(|message| CausalError::Unsupported { message })?,
            )
        } else {
            None
        })
    }

    /// Simultaneous bootstrap difference band, or the reason this design cannot publish one.
    fn difference_band(
        &self,
        columns: &SurvivalColumns,
        known_censoring: bool,
        endpoint: SurvivalEndpoint,
        assignment_counts: [usize; 2],
        ctx: &ExecutionContext,
    ) -> (Option<crate::result::SurvivalDifferenceBand>, Option<Arc<str>>) {
        let bootstrap_replicates = self.extra.0;
        if bootstrap_replicates == 0 {
            (None, None)
        } else if columns.entry.is_some() {
            (None, Some(Arc::from("simultaneous survival band does not cover delayed entry")))
        } else if known_censoring {
            (
                None,
                Some(Arc::from(
                    "simultaneous survival band does not cover caller-supplied censoring weights",
                )),
            )
        } else {
            match randomized_survival_bootstrap_difference_band(
                &columns.duration,
                &columns.event,
                &columns.treated,
                self.query.tau,
                endpoint,
                bootstrap_replicates,
                ctx.rng.master_seed() ^ 0x5A7A_BA4D,
            ) {
                Ok(band) => {
                    let band_claim = match self.query.functional {
                        SurvivalFunctional::SurvivalAndRmst => {
                            "simultaneous_95_survival_difference_band"
                        }
                        SurvivalFunctional::CumulativeIncidence { .. } => {
                            "simultaneous_95_cumulative_incidence_difference_band"
                        }
                    };
                    let band_support = (band.replicates_ok >= 399
                        && crate::support::license_if_graphless(
                            crate::support::GraphlessSupportKey {
                                family: "survival",
                                design: "two_arm_individual_randomized",
                                method: "arm_stratified_subject_bootstrap_supremum_band",
                                inference_claim: band_claim,
                            },
                            crate::support::GraphlessAssignmentSupport {
                                assignment_unit: "unit",
                                treated: assignment_counts[1],
                                control: assignment_counts[0],
                                rows: self.rows,
                                interval_95_published: true,
                                reported_intervals: 1,
                                ..Default::default()
                            },
                        )
                        .is_some())
                    .then_some(crate::support::CellStatus::Licensed);
                    (
                        Some(crate::result::SurvivalDifferenceBand {
                            times: band.times.into(),
                            difference: band.difference.into(),
                            lower: band.lower.into(),
                            upper: band.upper.into(),
                            replicates_ok: band.replicates_ok,
                            support_status: band_support,
                        }),
                        None,
                    )
                }
                Err(reason) => (None, Some(Arc::from(reason))),
            }
        }
    }

    /// Assembles the retained survival estimate from the point summary, bootstrap intervals, and band.
    fn survival_payload(
        &self,
        summary: RandomizedSurvivalSummary,
        intervals: Option<&SurvivalBootstrapIntervals>,
        minimum_censoring_survival: Option<f64>,
        assignment_counts: [usize; 2],
        band: (Option<crate::result::SurvivalDifferenceBand>, Option<Arc<str>>),
    ) -> crate::SurvivalEstimate {
        let (difference_band, band_unavailable_reason) = band;
        crate::SurvivalEstimate {
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
            uncertainty: Arc::from(if intervals.is_some() {
                "subject_stratified_percentile_bootstrap_pointwise_95"
            } else {
                "point_only_no_interval"
            }),
            rmst_difference_interval: intervals.and_then(|value| value.rmst_difference),
            difference_at_tau_interval: intervals.map(|value| value.difference_at_tau),
            bootstrap_replicates_requested: intervals
                .as_ref()
                .map(|value| value.replicates_requested),
            bootstrap_replicates_ok: intervals.map(|value| value.replicates_ok),
            assignment_counts,
            censoring_survival_provenance: minimum_censoring_survival
                .map(|_| Arc::from("caller_supplied_fixed_not_fitted_or_verified")),
            difference_band,
            band_unavailable_reason,
        }
    }

    /// Grants the graphless scalar license when the paired RMST and horizon intervals clear their support gate.
    fn license_survival_intervals(
        &self,
        result: &mut StudyResult,
        intervals: Option<&SurvivalBootstrapIntervals>,
        assignment_counts: [usize; 2],
    ) {
        // The paired RMST + horizon survival intervals carry a graphless scalar
        // license once their own route clears its support gate. Delayed entry,
        // fixed-known-G IPCW, and the combined route each have a distinct
        // (design, method) key and its own known-truth coverage row; cause-specific
        // cumulative incidence has no RMST companion and stays off-axis.
        if matches!(self.query.functional, SurvivalFunctional::SurvivalAndRmst)
            && intervals.is_some_and(|value| value.replicates_ok >= 299)
        {
            let delayed = self.query.delayed_entry.is_some();
            let ipcw = self.query.known_censoring.is_some();
            let design = if delayed {
                "delayed_entry_two_arm_individual_randomized"
            } else {
                "two_arm_individual_randomized"
            };
            let method = if ipcw {
                "arm_stratified_subject_bootstrap_ipcw_product_limit"
            } else {
                "arm_stratified_subject_bootstrap_product_limit"
            };
            if crate::support::license_if_graphless(
                crate::support::GraphlessSupportKey {
                    family: "survival",
                    design,
                    method,
                    inference_claim: "rmst_and_horizon_survival_pointwise_95_percentile_intervals",
                },
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    treated: assignment_counts[1],
                    control: assignment_counts[0],
                    rows: self.rows,
                    interval_95_published: intervals.is_some(),
                    reported_intervals: 2,
                    all_reported_intervals: true,
                    known_density: ipcw,
                    ..Default::default()
                },
            )
            .is_some()
            {
                result.support_status = Some(crate::support::CellStatus::Licensed);
            }
        }
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
            if query.known_censoring.is_some() && query.delayed_entry.is_some() {
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
    arena.set_derivation(functional, DerivationMeta::rule("randomized.survival_product_limit", Some(Arc::from("randomized assignment and independent observation identify arm event-time functionals"))));
    let estimand = IdentifiedEstimand::new(
        "randomized.survival_product_limit",
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push("randomized.survival_product_limit", "arm survival or cause-specific cumulative incidence is identified under random assignment and independent censoring/entry");
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
