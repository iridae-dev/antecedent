//! Retained nearest-neighbor contrast on pre-outcome histories, matched within calendar periods.
//!
//! The score is the mean of each unit's history columns. A treated unit is matched only to a
//! control of its own period, so a match never crosses a period, and a period with no control
//! leaves its treated units unmatched. The estimand is the sampling-weighted effect on the matched
//! treated units, conditional on the period and the history-mean score. The standard error is the
//! Abadie-Imbens form for one match with replacement; the interval allows serial dependence
//! between periods (Newey-West over the matched periods, normal at 97.5%) and is absent when
//! fewer than two periods hold a match. The route is unlicensed and its calibration is unmeasured. A posterior of this contrast does not exist (a
//! matching rule has no likelihood): Bayesian inference with the matching estimator is refused
//! with that premise and the next attempt. That next attempt is the estimator
//! `bayesian.period_history_outcome_model`: the posterior of a declared Gaussian outcome model
//! with period effects, the history-mean score and a treatment-by-score interaction centred at
//! the treated mean, reported as the model-based effect on the treated units (a different
//! estimand, with no donor pairing). It is unlicensed and uncalibrated as well.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
    require_built_in_model,
};
use super::{
    Arc, BayesianGCompWorkspace, CausalError, CausalExprArena, CausalQuery, DerivationMeta,
    DerivationTrace, Diagnostic, DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate,
    EstimatorId, ExecutionContext, ExprNode, GraphClass, IdentificationPerformanceRecord,
    IdentificationResult, IdentifiedEstimand, IdentifiedExecuteFinish, IdentifierId, InferenceMode,
    OutcomeExprId, OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation,
    RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData,
    bayesian_gcomp, finish_identified_execute_with_context, resolve_bayesian_prior_with_conflict,
};
use crate::analysis::matching_posterior_refusal::{MatchingSetting, matching_posterior_refusal};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};
use antecedent_estimate::history_match::{HistoryUnit, PeriodHistoryContrast};
use antecedent_estimate::history_outcome_model::prepare_history_outcome_model;

/// Uncertainty label of the history-matched contrast.
const UNCERTAINTY: &str = "abadie_imbens_and_newey_west_period_interval_calibration_unmeasured";

/// Frozen history match. The extra is the Bayesian configuration of the outcome-model estimator;
/// the matching contrast itself has no posterior and holds none.
pub(crate) type CheckedHistoryMatchOperation =
    RandomizedTrialOperation<antecedent_core::HistoryMatchQuery, Option<super::BayesianConfig>>;

impl RandomizedTrialQuery for antecedent_core::HistoryMatchQuery {
    const OPERATION: &'static str = "CheckedHistoryMatchOperation";
    const REFRESH_MISMATCH: &'static str =
        "history match refresh requires the prepared schema and row order";
}

impl CheckedHistoryMatchOperation {
    /// The design identification this operation records; the route lowering retains it.
    pub(crate) fn design_identification(
        query: &antecedent_core::HistoryMatchQuery,
    ) -> (IdentificationResult, IdentifiedEstimand) {
        match_identification(query)
    }

    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::HistoryMatch(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "history match operation requires HistoryMatchQuery".into(),
            });
        };
        query.validate()?;
        let bayesian = match &study.settings.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                if !physical.binding().names(
                    IdentifierId::RandomizedDesign,
                    EstimatorId::BayesianPeriodHistoryOutcomeModel,
                ) {
                    return Err(matching_posterior_refusal(MatchingSetting::WithinPeriodHistory));
                }
                require_built_in_model(config, "the Bayesian history outcome model")?;
                Some(config.clone())
            }
        };
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.settings.refute != RefuteSuite::None
            || study.settings.bootstrap_replicates != 0
            || !study.declared.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.declared.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "the history match reads one row per unit, with no graph, refuter or bootstrap",
            });
        }
        for id in history_columns(query) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        let estimator_id = if bayesian.is_some() {
            EstimatorId::BayesianPeriodHistoryOutcomeModel
        } else {
            EstimatorId::PeriodHistoryMatching
        };
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator_id)
        {
            return Err(CausalError::Compile {
                message: "history match query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = match_identification(query);
        Ok(Self::freeze(study, query, data, physical, identification, estimand, bayesian))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let columns = MatchColumns::read(&self.query, data)?;
        let units = columns.units()?;
        if let Some(config) = &self.extra {
            return self.execute_outcome_model(config, data, &units, ctx);
        }
        let estimate = PeriodHistoryContrast::try_new(self.query.caliper)
            .map_err(CausalError::from)?
            .contrast(&units)
            .map_err(CausalError::from)?;
        let mut differences = Vec::with_capacity(estimate.matched_queries.len());
        let mut pair_weights = Vec::with_capacity(estimate.matched_queries.len());
        for (query, donor) in estimate.matched_queries.iter().zip(&estimate.matched_donors) {
            let (Some(treated), Some(control), Some(weight)) = (
                columns.outcome.get(*query),
                columns.outcome.get(*donor),
                columns.weight.get(*query),
            ) else {
                return Err(CausalError::Compile {
                    message: "a matched row is outside the table".into(),
                });
            };
            differences.push(treated - control);
            pair_weights.push(*weight);
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
                Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::PeriodHistoryMatching),
                self.query.outcome,
                self.query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(vec![Diagnostic::new(
                "estimate.history_match.unlicensed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "unlicensed and uncalibrated route: the mean history carrying the confounding is \
                 the caller's assumption, a match never crosses a calendar period, the standard \
                 error is Abadie-Imbens for one match with replacement, the interval is the Newey-West \
                 interval over the matched calendar periods, and their calibration is unmeasured",
            )]),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.support_status = None;
        result.payload = crate::result::ResultPayload::HistoryMatch(crate::HistoryMatchEstimate {
            contrast: estimate.contrast,
            standard_error: estimate.standard_error,
            weight_total: estimate.weight_total,
            matched_queries: Arc::from(estimate.matched_queries.as_slice()),
            matched_donors: Arc::from(estimate.matched_donors.as_slice()),
            differences: Arc::from(differences.as_slice()),
            pair_weights: Arc::from(pair_weights.as_slice()),
            query_variances: Arc::from(estimate.query_variances.as_slice()),
            donor_variances: Arc::from(estimate.donor_variances.as_slice()),
            unmatched_treated: Arc::from(estimate.unmatched_treated.as_slice()),
            periods_matched: estimate.periods_matched,
            pair_periods: Arc::from(estimate.matched_periods.as_slice()),
            serial_standard_error: estimate.serial.as_ref().map(|s| s.standard_error),
            interval_lower: estimate.serial.as_ref().map(|s| s.lower),
            interval_upper: estimate.serial.as_ref().map(|s| s.upper),
            serial_bandwidth: estimate.serial.as_ref().map_or(0, |s| s.bandwidth),
            caliper: self.query.caliper,
            uncertainty: Arc::from(UNCERTAINTY),
        });
        Ok(result)
    }
}

impl CheckedHistoryMatchOperation {
    /// The posterior of the declared outcome model, reported as the model-based effect on the
    /// treated units of the periods that hold both arms.
    #[allow(
        clippy::too_many_lines,
        reason = "one fit: prepare the design, fit, summarize the effect draws, then assemble the result"
    )]
    fn execute_outcome_model(
        &self,
        config: &super::BayesianConfig,
        data: &TabularData,
        units: &[HistoryUnit],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = std::time::Instant::now();
        let problem = prepare_history_outcome_model(units).map_err(CausalError::from)?;
        let mut fitter = bayesian_gcomp(config, ctx);
        let (prior, _) =
            resolve_bayesian_prior_with_conflict(config, &problem.prepared, Some(ctx))?;
        fitter.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
        let posterior = fitter
            .fit(&problem.prepared, self.identification.status, &mut workspace, ctx)
            .map_err(CausalError::from)?;
        let column = posterior.effect_column().ok_or(CausalError::Compile {
            message: "the history outcome model produced no effect column".into(),
        })?;
        let draws = posterior.draws.column(column).map_err(CausalError::from)?.to_vec();
        let count = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / count;
        let sd = (draws.iter().map(|draw| (draw - mean).powi(2)).sum::<f64>()
            / (count - 1.0).max(1.0))
        .sqrt();
        let design = DesignPosterior {
            columns: vec![(effect_quantity("effect_on_treated"), draws)],
            point: mean,
            point_sd: sd,
            n_obs: problem.rows as u64,
            interval_withheld: false,
            backend: "bayesian.conjugate_gaussian.period_history_outcome_model",
            model: format!(
                "Gaussian outcome model with {} calendar-period effects, the history-mean score \
                 and a treatment-by-score interaction centred at the treated mean, fitted to {} \
                 units; the reported effect is the model-based contrast over the {} treated units, \
                 not a matching estimand",
                problem.periods, problem.rows, problem.treated_rows
            ),
            assumptions: vec![
                (
                    "bayesian.period_history_outcome_model.linear_gaussian",
                    "the outcome is linear in the history-mean score within a period, with a \
                     common treatment effect shift across periods and homoskedastic Gaussian \
                     residuals independent across units; the history mean carrying the \
                     confounding is the caller's assumption"
                        .to_string(),
                ),
                (
                    "bayesian.period_history_outcome_model.model_based_effect_on_treated",
                    "the reported effect is the model's contrast averaged over the treated \
                     units of the periods that hold both arms; it is a different estimand from \
                     the history-matched contrast and its interval is unlicensed with calibration \
                     unmeasured"
                        .to_string(),
                ),
            ],
            diagnostics: vec![Diagnostic::new(
                "estimate.history_outcome_model.unused_treated",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{} treated units sit in a period without a control and are not used",
                    problem.unused_treated.len()
                ),
            )],
            notes: vec![format!(
                "periods={} rows={} treated_rows={}",
                problem.periods, problem.rows, problem.treated_rows
            )],
            sampler: None,
        };
        finish_design_posterior_as(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            design,
            Strategy::new(
                IdentifierId::RandomizedDesign,
                EstimatorId::BayesianPeriodHistoryOutcomeModel,
            ),
        )
    }
}

impl Study {
    pub(super) fn execute_history_match(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedHistoryMatchOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

fn history_columns(
    query: &antecedent_core::HistoryMatchQuery,
) -> impl Iterator<Item = antecedent_core::VariableId> + '_ {
    [query.outcome, query.treated]
        .into_iter()
        .chain(query.history.iter().copied())
        .chain(query.period)
        .chain(query.weight)
}

/// The complete numeric columns the match reads.
struct MatchColumns {
    outcome: Vec<f64>,
    treated: Vec<f64>,
    history: Vec<Vec<f64>>,
    period: Vec<f64>,
    weight: Vec<f64>,
}

impl MatchColumns {
    fn read(
        query: &antecedent_core::HistoryMatchQuery,
        data: &TabularData,
    ) -> Result<Self, CausalError> {
        let read = |id| -> Result<Vec<f64>, CausalError> {
            match data.column(id).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) if column.validity.is_all_valid() => {
                    Ok(column.values.as_slice().to_vec())
                }
                _ => Err(CausalError::Unsupported {
                    message: "history match columns must be complete continuous data",
                }),
            }
        };
        let rows = data.row_count();
        Ok(Self {
            outcome: read(query.outcome)?,
            treated: read(query.treated)?,
            history: query.history.iter().map(|id| read(*id)).collect::<Result<_, _>>()?,
            period: query.period.map_or_else(|| Ok(vec![0.0; rows]), read)?,
            weight: query.weight.map_or_else(|| Ok(vec![1.0; rows]), read)?,
        })
    }

    /// One match unit per row: the history across columns, the outcome, the arm, the weight and
    /// the calendar period.
    fn units(&self) -> Result<Vec<HistoryUnit>, CausalError> {
        let mut units = Vec::with_capacity(self.outcome.len());
        for row in 0..self.outcome.len() {
            let history: Vec<f64> =
                self.history.iter().filter_map(|column| column.get(row).copied()).collect();
            let (Some(outcome), Some(treated), Some(weight), Some(period)) = (
                self.outcome.get(row),
                self.treated.get(row),
                self.weight.get(row),
                self.period.get(row),
            ) else {
                return Err(CausalError::Compile {
                    message: "history match columns differ in length".into(),
                });
            };
            let arm = if treated.to_bits() == 1.0_f64.to_bits() {
                true
            } else if treated.to_bits() == 0.0_f64.to_bits() {
                false
            } else {
                return Err(CausalError::Unsupported {
                    message: "the history match treatment column must be coded exactly 0 or 1",
                });
            };
            if period.fract() != 0.0 || !period.is_finite() {
                return Err(CausalError::Unsupported {
                    message: "the history match period column must hold whole numbers",
                });
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a whole-number period of a table row fits i64"
            )]
            let calendar = *period as i64;
            units.push(
                HistoryUnit::try_new(&history, *outcome, arm, *weight)
                    .map_err(CausalError::from)?
                    .in_period(calendar),
            );
        }
        Ok(units)
    }
}

/// Declared assumptions of the within-period history match.
fn match_assumptions() -> antecedent_core::AssumptionSet {
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
            "unconfoundedness_given_history_mean_within_period",
            "treatment is independent of the potential outcomes given the history-mean score within a calendar period; a mean history is a coarse sufficient statistic whose adequacy is the caller's assumption",
        ),
        (
            "score_overlap_within_period",
            "treated and control histories overlap in score within the period; an unmatched treated unit is dropped, not borrowed across periods",
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
    assumptions
}

fn match_identification(
    query: &antecedent_core::HistoryMatchQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = match_assumptions();
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
    let rule = "matching.period_history_contrast";
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            rule,
            Some(Arc::from("unconfoundedness given the history-mean score within a period identifies the effect on the matched treated units")),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(
        rule,
        "the effect on matched treated units is identified within a period under declared unconfoundedness given the history-mean score, overlap and consistency",
    );
    let identification = IdentificationResult::identified(
        CausalQuery::HistoryMatch(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
