//! Retained two-period panel DiD execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct CheckedPanelDidOperation {
    query: antecedent_core::PanelDidQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedPanelDidOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedPanelDidOperation")
            .field("query", &self.query)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl CheckedPanelDidOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PanelDid(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "panel DiD operation requires PanelDidQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || query.treated.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "panel DiD currently accepts only the graphless two-period panel query with point estimate and cluster standard error",
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (identification, estimand) = panel_did_identification(query);
        Ok(Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported {
                message: "panel DiD refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "panel DiD outcome must be continuous",
                });
            }
        };
        let mut subjects: BTreeMap<&str, (Option<f64>, Option<f64>, Option<bool>, Option<&str>)> =
            BTreeMap::new();
        for i in 0..y.len() {
            if !y[i].is_finite() {
                return Err(CausalError::Unsupported {
                    message: "panel DiD requires complete finite outcomes",
                });
            }
            let subject = self.query.subjects[i].as_ref();
            let entry = subjects.entry(subject).or_insert((None, None, None, None));
            if entry.2.is_some_and(|v| v != self.query.treated[i]) {
                return Err(CausalError::Compile {
                    message: "treatment must be stable within subject".into(),
                });
            }
            entry.2 = Some(self.query.treated[i]);
            if entry.3.is_some_and(|v| v != self.query.clusters[i].as_ref()) {
                return Err(CausalError::Compile {
                    message: "cluster must be stable within subject".into(),
                });
            }
            entry.3 = Some(self.query.clusters[i].as_ref());
            let slot = if self.query.post[i] { &mut entry.1 } else { &mut entry.0 };
            if slot.replace(y[i]).is_some() {
                return Err(CausalError::Compile {
                    message: "each subject must have exactly one pre and one post observation"
                        .into(),
                });
            }
        }
        let mut changes = Vec::with_capacity(subjects.len());
        let mut sums = [0.0; 2];
        let mut counts = [0usize; 2];
        for (pre, post, treated, cluster) in subjects.values() {
            let (Some(pre), Some(post), Some(treated), Some(cluster)) =
                (pre, post, treated, cluster)
            else {
                return Err(CausalError::Compile {
                    message: "panel DiD requires a balanced two-period panel".into(),
                });
            };
            let change = post - pre;
            let group = usize::from(*treated);
            sums[group] += change;
            counts[group] += 1;
            changes.push((*treated, *cluster, change));
        }
        if counts.contains(&0) {
            return Err(CausalError::Compile {
                message: "panel DiD requires treated and comparison subjects".into(),
            });
        }
        let means = [sums[0] / counts[0] as f64, sums[1] / counts[1] as f64];
        let effect = means[1] - means[0];
        let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
        let mut clusters: [BTreeSet<&str>; 2] = Default::default();
        for (treated, cluster, change) in changes {
            let g = usize::from(treated);
            clusters[g].insert(cluster);
            *scores.entry(cluster).or_default() +=
                (if treated { 1.0 } else { -1.0 }) * (change - means[g]) / counts[g] as f64;
        }
        if clusters.iter().any(|set| set.len() < 2) {
            return Err(CausalError::Unsupported {
                message: "cluster SE requires at least two clusters in each group",
            });
        }
        let g = scores.len();
        let variance = g as f64 / (g - 1) as f64 * scores.values().map(|v| v * v).sum::<f64>();
        let identification = self.identification.clone();
        let assumptions = identification.required_assumptions.clone();
        let estimate = EffectEstimate::new(
            effect,
            variance.sqrt(),
            assumptions,
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification,
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.panel_did.cluster_se",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "two-period subject change-score DiD; cluster score sandwich SE; normal intervals are not reported",
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.panel_did = Some(crate::PanelDidEstimate {
            effect,
            standard_error: variance.sqrt(),
            treated_subjects: counts[1],
            comparison_subjects: counts[0],
            clusters: g,
            uncertainty: Arc::from("cluster_robust_standard_error_no_interval"),
        });
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_panel_did(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPanelDidOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn panel_did_identification(
    query: &antecedent_core::PanelDidQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for assumption in [antecedent_core::Assumption::Consistency, antecedent_core::Assumption::NoInterference] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption,
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    for (id, description) in [
        (
            "parallel_trends",
            "in the absence of treatment, treated and comparison groups would have had equal mean outcome changes",
        ),
        ("no_anticipation", "treatment does not affect pre-period outcomes"),
        ("stable_treatment", "treatment assignment is constant within subject"),
        (
            "balanced_panel",
            "each subject contributes exactly one observed outcome in each of two periods",
        ),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(outcomes, empty, intervention, antecedent_expr::DomainRef::Observational);
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome), distribution,
    });
    arena.set_derivation(functional, antecedent_expr::DerivationMeta::rule(
        "did.panel_change_score", Some(Arc::from("typed PanelDidQuery binds the observed group, period, subject, and cluster design"))));
    let estimand = IdentifiedEstimand::new("did.panel_change_score", Arc::from([]), Arc::from([]), Arc::from([]), functional, None);
    let mut derivation = DerivationTrace::default();
    derivation.push("did.panel_change_score", "the balanced-panel difference in subject-level changes is identified under parallel trends and no anticipation");
    let identification = IdentificationResult::identified(
        CausalQuery::PanelDid(query.clone()), vec![estimand.clone()], arena, derivation, assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
