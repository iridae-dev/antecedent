//! Context adapters for advisory, fit-free data diagnostics.
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "The facade consistently returns the established CausalError by value."
)]

use std::sync::Arc;

use antecedent_core::{
    CanonicalEncode, CanonicalWriter, CausalQuery, ExecutionContext, Intervention,
};
use antecedent_data::{TableView, TabularData};
pub use antecedent_validate::data_diagnostics::{
    DataDiagnosticCheck, DataDiagnosticColumn, DataDiagnosticContext, DataDiagnosticFinding,
    DataDiagnostics, DataDiagnosticsError, DataDiagnosticsReport,
};

use super::{builder::DataInput, execute::Study, prepared::PreparedStudy};
use crate::CausalError;

thread_local! {
    static BATCH_PROFILES: std::cell::RefCell<Option<std::collections::BTreeMap<String, Arc<DataDiagnosticsReport>>>> = const { std::cell::RefCell::new(None) };
}

pub(super) struct ProfileScope(bool);
impl Drop for ProfileScope {
    fn drop(&mut self) {
        if self.0 {
            BATCH_PROFILES.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

pub(super) fn batch_profiles() -> ProfileScope {
    ProfileScope(BATCH_PROFILES.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            false
        } else {
            *slot = Some(std::collections::BTreeMap::new());
            true
        }
    }))
}

/// Convert explicit diagnostic failures to existing operational error classes.
pub(crate) fn causal(error: DataDiagnosticsError) -> CausalError {
    match error {
        DataDiagnosticsError::Data(e) => CausalError::Data(e),
        DataDiagnosticsError::Policy(message) => CausalError::Compile { message },
        DataDiagnosticsError::Cancelled => {
            CausalError::Cancelled { stage: super::stage::STAGE_VALIDATE }
        }
        DataDiagnosticsError::Resource(message) => CausalError::Resource { message },
    }
}

/// Inspect data without requiring preparation or successful identification.
///
/// # Errors
/// Invalid policy or context, storage access, resource admission or cancellation.
pub fn diagnose_data(
    data: &TabularData,
    policy: &DataDiagnostics,
    context: &DataDiagnosticContext,
    ctx: &ExecutionContext,
) -> Result<DataDiagnosticsReport, CausalError> {
    antecedent_validate::data_diagnostics::diagnose_data(data, policy, context, ctx).map_err(causal)
}

fn table(study: &Study) -> Option<TabularData> {
    match &study.data {
        DataInput::Tabular(data) => Some(data.clone()),
        DataInput::Temporal(series) | DataInput::Event(series) => {
            Some(TabularData::new(series.storage().clone()))
        }
        DataInput::Panel(_) | DataInput::MultiEnv(_) => None,
    }
}

fn context(study: &Study, data: &TabularData) -> DataDiagnosticContext {
    let mut context = DataDiagnosticContext::default();
    let mut writer = CanonicalWriter::new("antecedent.data_diagnostics.context.v1");
    study.query.encode(&mut writer);
    let mut role = |id, name: &str| {
        if let Ok(variable) = data.schema().get(id) {
            context.roles.insert(variable.name.to_string(), name.into());
        }
    };
    match &study.query {
        CausalQuery::AverageEffect(q) => {
            role(q.treatment, "treatment");
            role(q.outcome, "outcome");
        }
        CausalQuery::ConditionalEffect(q) => {
            role(q.inner.treatment, "treatment");
            role(q.inner.outcome, "outcome");
        }
        CausalQuery::TemporalEffect(q) => {
            role(q.treatment, "treatment");
            role(q.outcome, "outcome");
        }
        CausalQuery::Response(q) => {
            for id in q.functional.treatment_ids() {
                role(id, "treatment");
            }
            for id in q.functional.outcome_ids() {
                role(id, "outcome");
            }
        }
        CausalQuery::Mediation(q) => {
            role(q.treatment, "treatment");
            role(q.outcome, "outcome");
        }
        _ => {}
    }
    if let Some(cache) = study.static_cache() {
        for id in cache.estimand.adjustment_set.iter() {
            if let Ok(v) = data.schema().get(*id) {
                context.roles.entry(v.name.to_string()).or_insert_with(|| "adjustment".into());
            }
        }
    }
    if let Ok(input) = super::preflight::input_of(study) {
        for id in &input.treatments {
            if let Ok(v) = data.schema().get(*id) {
                context.roles.insert(v.name.to_string(), "treatment".into());
            }
        }
        if let Ok(v) = data.schema().get(input.outcome) {
            context.roles.insert(v.name.to_string(), "outcome".into());
        }
    }
    let contrast = match &study.query {
        CausalQuery::AverageEffect(query) => Some(query),
        CausalQuery::ConditionalEffect(query) => Some(&query.inner),
        _ => None,
    };
    if let Some(q) = contrast {
        if let (Intervention::Set { value: control, .. }, Intervention::Set { value: active, .. }) =
            (&q.control, &q.active)
        {
            if let (Some(control), Some(active), Ok(variable)) =
                (control.as_f64(), active.as_f64(), data.schema().get(q.treatment))
            {
                context.contrasts.insert(variable.name.to_string(), vec![control, active]);
            }
        }
    }
    writer.text(&serde_json::to_string(&context.roles).expect("roles"));
    context.plan_identity = writer.finish_hex();
    context
}

#[allow(
    clippy::too_many_lines,
    reason = "The adapter records all preflight applicability and operational outcomes together."
)]
pub(crate) fn evaluate(
    study: &Study,
    policy: &DataDiagnostics,
    ctx: &ExecutionContext,
    automatic: bool,
) -> Result<Arc<DataDiagnosticsReport>, CausalError> {
    policy.validate().map_err(causal)?;
    let Some(data) = table(study) else {
        return Ok(Arc::new(DataDiagnosticsReport::unavailable(
            policy.clone(),
            DataDiagnosticContext::default(),
            "panel and multi-environment inputs require per-table diagnostics",
        )));
    };
    let context = context(study, &data);
    let key = format!(
        "{}:{:?}:{}",
        antecedent_validate::staleness::snapshot_identity(&data),
        data.schema().variables().iter().map(|v| (&v.name, &v.value_type)).collect::<Vec<_>>(),
        serde_json::to_string(policy).expect("validated policy")
    );
    let reused = BATCH_PROFILES
        .with(|slot| slot.borrow().as_ref().and_then(|cache| cache.get(&key).cloned()));
    let evaluated = match reused {
        Some(profile) => Ok((*profile).clone()),
        None => diagnose_data(&data, policy, &DataDiagnosticContext::default(), ctx),
    };
    let mut report = match evaluated {
        Ok(report) => report,
        Err(CausalError::Resource { .. }) if automatic => {
            let mut report = DataDiagnosticsReport::unavailable(
                policy.clone(),
                context,
                "automatic diagnostic resource admission refused",
            );
            report.snapshot_identity =
                Some(antecedent_validate::staleness::snapshot_identity(&data));
            report.rows_total = data.row_count();
            report.seal();
            return Ok(Arc::new(report));
        }
        Err(error) => return Err(error),
    };
    BATCH_PROFILES.with(|slot| {
        if let Some(cache) = slot.borrow_mut().as_mut() {
            cache.entry(key).or_insert_with(|| Arc::new(report.clone()));
        }
    });
    antecedent_validate::data_diagnostics::bind_context(&data, &mut report, &context, ctx)
        .map_err(causal)?;
    report.checks.retain(|check| check.code != "data.relationships");
    if !policy.relationships {
        report.check(
            "data.relationships",
            None,
            "not_evaluated",
            Some("disabled by suite"),
            0,
            false,
        );
    } else if let Ok(input) = super::preflight::input_of(study) {
        let scalar = input
            .treatments
            .iter()
            .chain(std::iter::once(&input.outcome))
            .chain(input.adjustment.iter())
            .all(|id| {
                matches!(
                    data.column(*id),
                    Ok(antecedent_data::ColumnView::Float64(_)
                        | antecedent_data::ColumnView::Int64(_)
                        | antecedent_data::ColumnView::Boolean(_))
                )
            });
        if !scalar {
            report.check(
                "data.relationships",
                None,
                "not_applicable",
                Some("existing preflight requires numeric or boolean scalar columns"),
                0,
                false,
            );
            report.seal();
            return Ok(Arc::new(report));
        }
        let bounded = policy.max_sample_rows.is_some_and(|limit| data.row_count() > limit)
            || (policy.max_sample_rows.is_some()
                && input.adjustment.len() + 1 > policy.max_relationship_columns);
        if bounded {
            report.check(
                "data.relationships",
                None,
                "not_evaluated",
                Some("full design exceeds bounded preflight limits"),
                0,
                false,
            );
        } else {
            match super::preflight::preflight_design(&input, ctx) {
                Ok(preflight) => {
                    for finding in preflight.findings {
                        let observation = finding
                            .measure
                            .map(|value| {
                                std::collections::BTreeMap::from([("measure".into(), value)])
                            })
                            .unwrap_or_default();
                        report.finding(finding.code, match finding.severity {
                            super::preflight::FindingSeverity::Review => "review",
                            super::preflight::FindingSeverity::Blocking => "blocking",
                        }, finding.columns, observation, finding.threshold.map_or_else(|| "existing preflight execution requirement".into(), |v| format!("existing preflight threshold {v}")),
                            &finding.detail, "Inspect the preflight report and declared design; no automatic changes are made.", false);
                    }
                    report.check(
                        "data.relationships",
                        None,
                        "evaluated",
                        None,
                        data.row_count(),
                        false,
                    );
                }
                Err(CausalError::Resource { .. }) if automatic => report.check(
                    "data.relationships",
                    None,
                    "not_evaluated",
                    Some("automatic preflight resource admission refused"),
                    0,
                    false,
                ),
                Err(error) => return Err(error),
            }
        }
    } else {
        report.check(
            "data.relationships",
            None,
            "not_applicable",
            Some("existing preflight does not support this plan"),
            0,
            false,
        );
    }
    report.seal();
    Ok(Arc::new(report))
}

impl PreparedStudy {
    /// Automatically collected report, absent only when collection was disabled.
    #[must_use]
    pub fn data_diagnostics(&self) -> Option<&DataDiagnosticsReport> {
        self.state.data_diagnostics.as_deref()
    }

    /// Explicit fit-free evaluation, independent of the automatic collection setting.
    ///
    /// # Errors
    /// Invalid policy, storage access, resource admission or cancellation.
    pub fn diagnose_data(
        &self,
        policy: &DataDiagnostics,
        ctx: &ExecutionContext,
    ) -> Result<DataDiagnosticsReport, CausalError> {
        evaluate(self.study(), policy, ctx, false).map(|report| (*report).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_graph::{Dag, DenseNodeId};

    fn study() -> Study {
        let t: Vec<f64> = (0..200).map(|i| f64::from(i % 2)).collect();
        let y: Vec<f64> = (0..200).map(f64::from).collect();
        let data =
            TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .estimator(crate::EstimatorId::LinearAdjustmentAte)
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    #[test]
    fn automatic_resource_refusal_is_coverage_and_explicit_refusal_is_error() {
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(1));
        let study = study();
        let report = evaluate(&study, &DataDiagnostics::default(), &ctx, true).unwrap();
        assert!(report.findings.is_empty());
        assert!(report.rows_complete.is_none());
        assert!(report.checks.iter().all(|check| check.status == "not_evaluated"));
        assert!(matches!(
            evaluate(&study, &DataDiagnostics::default(), &ctx, false),
            Err(CausalError::Resource { .. })
        ));
        ctx.cancellation.cancel();
        assert!(matches!(
            evaluate(&study, &DataDiagnostics::default(), &ctx, true),
            Err(CausalError::Cancelled { .. })
        ));
    }

    #[test]
    fn scoped_batch_cache_reuses_only_neutral_profiles_and_clears_afterward() {
        let ctx = ExecutionContext::for_tests(1);
        {
            let _scope = batch_profiles();
            let study = study();
            let a = evaluate(&study, &DataDiagnostics::default(), &ctx, true).unwrap();
            let b = evaluate(&study, &DataDiagnostics::default(), &ctx, true).unwrap();
            assert_eq!(a.identity, b.identity);
            BATCH_PROFILES.with(|slot| {
                let borrowed = slot.borrow();
                let cache = borrowed.as_ref().unwrap();
                assert_eq!(cache.len(), 1);
                assert!(cache.values().next().unwrap().context.roles.is_empty());
            });
        }
        BATCH_PROFILES.with(|slot| assert!(slot.borrow().is_none()));
    }
}

impl super::batch::PreparedBatch {
    /// Automatically collected member reports in declared query order.
    #[must_use]
    pub fn data_diagnostics(&self) -> Vec<Option<&DataDiagnosticsReport>> {
        self.plans().iter().map(PreparedStudy::data_diagnostics).collect()
    }

    /// Explicit fit-free member evaluation with compatible table profiles shared.
    ///
    /// # Errors
    /// Invalid policy, storage access, resource admission or cancellation.
    pub fn diagnose_data(
        &self,
        policy: &DataDiagnostics,
        ctx: &ExecutionContext,
    ) -> Result<Vec<DataDiagnosticsReport>, CausalError> {
        let _profiles = batch_profiles();
        self.plans().iter().map(|plan| plan.diagnose_data(policy, ctx)).collect()
    }
}
