//! Retained nearest-neighbor contrast on pre-outcome histories, matched within calendar periods.
//!
//! The score is the mean of each unit's history columns. A treated unit is matched only to a
//! control of its own period, so a match never crosses a period, and a period with no control
//! leaves its treated units unmatched. The estimand is the sampling-weighted effect on the matched
//! treated units, conditional on the period and the history-mean score. The standard error is the
//! Abadie-Imbens form for one match with replacement; no interval is reported, the route is
//! unlicensed and its calibration is unmeasured. A posterior of this contrast does not exist (a
//! matching rule has no likelihood): Bayesian inference is refused with that premise and the next
//! attempt.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteFinish, IdentifierId, InferenceMode, OutcomeExprId,
    OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation, RandomizedTrialQuery,
    RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData,
    finish_identified_execute_with_context,
};
use crate::analysis::matching_posterior_refusal::{MatchingSetting, matching_posterior_refusal};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};
use antecedent_estimate::history_match::{HistoryUnit, PeriodHistoryContrast};

/// Uncertainty label of the history-matched contrast.
const UNCERTAINTY: &str = "abadie_imbens_standard_error_calibration_unmeasured_no_interval";

/// Frozen history match; it holds no Bayesian configuration because it has no posterior.
pub(crate) type CheckedHistoryMatchOperation =
    RandomizedTrialOperation<antecedent_core::HistoryMatchQuery, ()>;

impl RandomizedTrialQuery for antecedent_core::HistoryMatchQuery {
    const OPERATION: &'static str = "CheckedHistoryMatchOperation";
    const REFRESH_MISMATCH: &'static str =
        "history match refresh requires the prepared schema and row order";
}

impl CheckedHistoryMatchOperation {
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
        if matches!(study.inference, InferenceMode::Bayesian(_)) {
            return Err(matching_posterior_refusal(MatchingSetting::WithinPeriodHistory));
        }
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "the history match reads one row per unit, with no graph, refuter or bootstrap",
            });
        }
        for id in history_columns(query) {
            data.schema().get(id).map_err(CausalError::rejected)?;
        }
        if physical.logical.query != study.query
            || !physical
                .binding()
                .names(IdentifierId::RandomizedDesign, EstimatorId::PeriodHistoryMatching)
        {
            return Err(CausalError::Compile {
                message: "history match query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = match_identification(query);
        Ok(Self::freeze(study, query, data, physical, identification, estimand, ()))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let columns = MatchColumns::read(&self.query, data)?;
        let units = columns.units()?;
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
                 error is Abadie-Imbens for one match with replacement and its calibration is \
                 unmeasured, and no interval is reported",
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
            unmatched_treated: Arc::from(estimate.unmatched_treated.as_slice()),
            periods_matched: estimate.periods_matched,
            caliper: self.query.caliper,
            uncertainty: Arc::from(UNCERTAINTY),
        });
        Ok(result)
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
