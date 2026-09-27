//! Retained graphless randomized survival execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    SurvivalFunctional,
};
use antecedent_estimate::survival::{SurvivalEndpoint, randomized_survival_bootstrap_difference_band, randomized_survival_bootstrap_intervals, randomized_survival_ipcw_summary_with_entry, randomized_survival_summary};

#[derive(Clone)]
pub(crate) struct CheckedSurvivalOperation {
    query: antecedent_core::SurvivalQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
    bootstrap_replicates: u32,
}

impl std::fmt::Debug for CheckedSurvivalOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedSurvivalOperation")
            .field("query", &self.query)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl CheckedSurvivalOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Survival(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "survival operation requires SurvivalQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "survival requires a graphless randomized trial, an explicit observation assumption, and frequentist execution",
            });
        }
        for id in
            [Some(query.duration), Some(query.event), Some(query.treatment), query.delayed_entry]
                .into_iter()
                .flatten()
        {
            data.schema().get(id).map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
        if let Some(known) = &query.known_censoring {
            for id in known.columns.iter().copied() {
                data.schema().get(id).map_err(|e| CausalError::Compile { message: e.to_string() })?;
            }
        }
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::RandomizedDesign.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::RandomizedSurvivalProductLimit.as_str())
        {
            return Err(CausalError::Compile {
                message: "survival query differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) = survival_identification(query);
        Ok(Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
            bootstrap_replicates: study.bootstrap_replicates,
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported {
                message: "survival refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let duration = numeric_column(data, self.query.duration)?;
        let event_raw = numeric_column(data, self.query.event)?;
        let treatment_raw = numeric_column(data, self.query.treatment)?;
        let entry = self.query.delayed_entry.map(|id| numeric_column(data, id)).transpose()?;
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
        let endpoint = match self.query.functional {
            SurvivalFunctional::SurvivalAndRmst => SurvivalEndpoint::Survival,
            SurvivalFunctional::CumulativeIncidence { target_cause } => {
                SurvivalEndpoint::CumulativeIncidence { target_cause }
            }
        };
        let mut censoring_grid = None;
        let (summary, minimum_censoring_survival) = if let Some(known) = &self.query.known_censoring {
            let columns = known.columns.iter().map(|id| numeric_column(data, *id)).collect::<Result<Vec<_>, _>>()?;
            let mut probabilities = Vec::with_capacity(self.rows * columns.len());
            for row in 0..self.rows {
                probabilities.extend(columns.iter().map(|column| column[row]));
            }
            censoring_grid = Some((known.times.as_ref(), probabilities, known.minimum_probability));
            let (_, probabilities, _) = censoring_grid.as_ref().expect("just set");
            let (summary, minimum) = randomized_survival_ipcw_summary_with_entry(
                &duration, &event, &treated, entry.as_deref(), &known.times, probabilities,
                self.query.tau, known.minimum_probability, endpoint,
            ).map_err(|message| CausalError::Unsupported { message })?;
            (summary, Some(minimum))
        } else {
            (randomized_survival_summary(
                &duration, &event, &treated, entry.as_deref(), self.query.tau, endpoint,
            ).map_err(|message| CausalError::Unsupported { message })?, None)
        };
        let intervals = if self.bootstrap_replicates > 0 {
            Some(randomized_survival_bootstrap_intervals(
                &duration, &event, &treated, entry.as_deref(),
                censoring_grid.as_ref().map(|(times, probabilities, floor)| (*times, probabilities.as_slice(), *floor)),
                self.query.tau, endpoint, self.bootstrap_replicates, ctx.rng.master_seed(),
            ).map_err(|message| CausalError::Unsupported { message })?)
        } else { None };
        let (difference_band, band_unavailable_reason) = if self.bootstrap_replicates == 0 {
            (None, None)
        } else if entry.is_some() {
            (None, Some(Arc::from("simultaneous survival band does not cover delayed entry")))
        } else if censoring_grid.is_some() {
            (None, Some(Arc::from("simultaneous survival band does not cover caller-supplied censoring weights")))
        } else {
            match randomized_survival_bootstrap_difference_band(
                &duration, &event, &treated, self.query.tau, endpoint,
                self.bootstrap_replicates, ctx.rng.master_seed() ^ 0x5A7A_BA4D,
            ) {
                Ok(band) => (Some(crate::result::SurvivalDifferenceBand {
                    times: band.times.into(), difference: band.difference.into(),
                    lower: band.lower.into(), upper: band.upper.into(),
                    replicates_ok: band.replicates_ok,
                }), None),
                Err(reason) => (None, Some(Arc::from(reason))),
            }
        };
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate: EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedSurvivalProductLimit,
                treatment: self.query.treatment,
                outcome: self.query.duration,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
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
                ))).collect(),
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: 0,
                bootstrap_replicates_ok: intervals.as_ref().map(|value| value.replicates_ok),
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.survival = Some(crate::SurvivalEstimate {
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
            uncertainty: Arc::from(if intervals.is_some() { "subject_stratified_percentile_bootstrap_pointwise_95" } else { "point_only_no_interval" }),
            rmst_difference_interval: intervals.as_ref().and_then(|value| value.rmst_difference),
            difference_at_tau_interval: intervals.as_ref().map(|value| value.difference_at_tau),
            bootstrap_replicates_requested: intervals.as_ref().map(|value| value.replicates_requested),
            bootstrap_replicates_ok: intervals.as_ref().map(|value| value.replicates_ok),
            censoring_survival_provenance: minimum_censoring_survival.map(|_| Arc::from("caller_supplied_fixed_not_fitted_or_verified")),
            difference_band,
            band_unavailable_reason,
        });
        Ok(result)
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
        physical: &PhysicalExecutionPlan,
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
