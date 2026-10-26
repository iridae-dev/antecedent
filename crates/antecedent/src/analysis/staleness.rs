//! Staleness and drift diagnostics of a prepared study or an exported baseline.
//!
//! [`PreparedStudy::staleness`] compares new data with the data the study retains (the data
//! it was prepared on, or the data of its last successful refresh) under a declared
//! [`StalenessPolicy`], and returns a [`StalenessReport`]: typed findings, the dependency keys
//! they change, the stages selective recalculation would recompute, and one
//! [`Recommendation`] (`reuse`, `refresh`, `reidentify` or `refuse`) with the attempt to make
//! next. [`PreparedStudy::staleness_baseline`] exports the summary the comparison reads, and
//! [`assess_artifact_staleness`] runs the same comparison from those bytes alone, without the
//! producer or its rows.
//!
//! The diagnostic decides and explains; it changes nothing. A host acts on the recommendation
//! by calling [`PreparedStudy::refresh`] or by preparing again; no scheduler or alerting
//! service is part of this module. See `docs/guides/staleness.md`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalEncode, CanonicalWriter, CausalQuery, ExecutionContext};
use antecedent_data::TabularData;
pub use antecedent_validate::staleness::{
    FindingKind, Multiplicity, Recommendation, StalenessBaseline, StalenessError, StalenessFinding,
    StalenessPlan, StalenessPolicy, StalenessReport, dependency_class, snapshot_identity,
};
use serde_json::{Map, Value, json};

use super::builder::DataInput;
use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::inference::InferenceMode;

const PLAN_TAG: &str = "antecedent.staleness.plan.v1";

/// The causal error of a staleness diagnostic that could not run.
fn causal(error: StalenessError) -> CausalError {
    match error {
        StalenessError::Data(error) => CausalError::Data(error),
        StalenessError::Unsupported(message) => CausalError::Unsupported { message },
        StalenessError::Cancelled => CausalError::Cancelled { stage: super::stage::STAGE_VALIDATE },
        StalenessError::Admission { .. } | StalenessError::ArtifactAdmission { .. } => {
            CausalError::Resource { message: error.to_string() }
        }
        other => CausalError::Compile { message: other.to_string() },
    }
}

impl PreparedStudy {
    /// The roles a staleness baseline of this study records: the query's treatment and
    /// outcome, the adjustment set of the cached identification, and whether the study is
    /// Bayesian.
    #[must_use]
    pub fn staleness_plan(&self) -> StalenessPlan {
        let study = self.study();
        let schema = self.schema();
        let name = |id: antecedent_core::VariableId| {
            schema.get(id).ok().map(|variable| variable.name.to_string())
        };
        let (treatment, outcome) = match &study.query {
            CausalQuery::AverageEffect(query) => (name(query.treatment), name(query.outcome)),
            CausalQuery::ConditionalEffect(query) => {
                (name(query.inner.treatment), name(query.inner.outcome))
            }
            CausalQuery::Mediation(query) => (name(query.treatment), name(query.outcome)),
            _ => (None, None),
        };
        let adjustment: Vec<String> = study
            .static_cache()
            .map(|cache| cache.estimand.adjustment_set.iter().filter_map(|id| name(*id)).collect())
            .unwrap_or_default();
        let bayesian = matches!(study.inference, InferenceMode::Bayesian(_));
        let plan_identity =
            self.contract().ok().and_then(|contract| contract.identities.program).map_or_else(
                || {
                    let mut writer = CanonicalWriter::new(PLAN_TAG);
                    study.query.encode(&mut writer);
                    for variable in schema.variables() {
                        writer.text(&variable.name);
                    }
                    writer.opt_text(treatment.as_deref()).opt_text(outcome.as_deref());
                    writer.texts(&adjustment).word(u64::from(bayesian));
                    writer.finish_hex()
                },
                |program| program.to_hex(),
            );
        StalenessPlan { plan_identity, treatment, outcome, adjustment, bayesian }
    }

    /// The data snapshot the study retains, as a table.
    fn retained_table(&self) -> Result<TabularData, CausalError> {
        match &self.study().data {
            DataInput::Tabular(data) => Ok(data.clone()),
            DataInput::Temporal(series) | DataInput::Event(series) => {
                Ok(TabularData::new(series.storage().clone()))
            }
            DataInput::MultiEnv(_) | DataInput::Panel(_) => Err(CausalError::Unsupported {
                message: "staleness diagnostics compare tabular or series snapshots; a panel or \
                          multi-environment study is assessed per unit or environment table",
            }),
        }
    }

    /// The sealed summary of the data this study retains, under `policy`.
    ///
    /// # Errors
    ///
    /// An invalid policy, a panel or multi-environment study, a refused admission lease, or
    /// cancellation.
    pub fn staleness_baseline(
        &self,
        policy: &StalenessPolicy,
        ctx: &ExecutionContext,
    ) -> Result<StalenessBaseline, CausalError> {
        // Estimator sampling weights are a population declaration, separate from
        // storage weights. The unweighted profile cannot certify that population.
        if self.study().estimator_spec.as_deref().is_some_and(|spec| {
            use crate::estimator_spec::EstimatorSpec;
            match spec {
                EstimatorSpec::LinearAdjustmentAte(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::PropensityWeighting(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::PropensityMatching(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::PropensityStratification(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::Aipw(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::DistanceMatching(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::GlmAdjustment(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::Dml(cfg) => cfg.sampling_weights.is_some(),
                EstimatorSpec::DrLearner(cfg) => cfg.sampling_weights.is_some(),
                _ => false,
            }
        }) {
            return Err(causal(StalenessError::Unsupported(
                "sampling-weighted populations require a declared weighted staleness diagnostic",
            )));
        }
        let data = self.retained_table()?;
        StalenessBaseline::build(&data, &self.staleness_plan(), policy, ctx).map_err(causal)
    }

    /// Compare `new_data` with the data this study retains, under `policy`.
    ///
    /// The report's recommendation is the most severe one any exceeded diagnostic forces, and
    /// its invalidated stages are those selective recalculation recomputes when the named keys
    /// change. The study is not modified.
    ///
    /// # Errors
    ///
    /// An invalid policy, a panel or multi-environment study, an unreadable column, a refused
    /// admission lease, or cancellation. Data the plan cannot be applied to is a
    /// [`Recommendation::Refuse`] report, not an error.
    pub fn staleness(
        &self,
        new_data: &TabularData,
        policy: &StalenessPolicy,
        ctx: &ExecutionContext,
    ) -> Result<StalenessReport, CausalError> {
        self.staleness_baseline(policy, ctx)?.assess(new_data, policy, ctx).map_err(causal)
    }
}

/// Compare `new_data` with an exported staleness baseline (`artifact`), without the producer.
///
/// # Errors
///
/// The artifact does not verify, the policy is invalid, a column is unreadable, the admission
/// pool refuses the lease, or the context is cancelled.
pub fn assess_artifact_staleness(
    artifact: &[u8],
    new_data: &TabularData,
    policy: &StalenessPolicy,
    ctx: &ExecutionContext,
) -> Result<StalenessReport, CausalError> {
    antecedent_validate::staleness::assess_exported_baseline(artifact, new_data, policy, ctx)
        .map_err(causal)
}

fn number(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// A finding as JSON. Non-finite statistics are `null`.
#[must_use]
pub fn staleness_finding_json(finding: &StalenessFinding) -> Value {
    let auxiliary: Map<String, Value> =
        finding.auxiliary.iter().map(|(name, value)| (name.clone(), number(*value))).collect();
    json!({
        "kind": finding.kind.label(),
        "subject": finding.subject,
        "statistic_name": finding.statistic_name,
        "statistic": number(finding.statistic),
        "threshold_name": finding.threshold_name,
        "threshold": number(finding.threshold),
        "p_value": finding.p_value.map(number),
        "adjusted_p_value": finding.adjusted_p_value.map(number),
        "exceeded": finding.exceeded,
        "forces": finding.forces.label(),
        "keys": finding.keys.iter().map(|k| k.label()).collect::<Vec<_>>(),
        "auxiliary": auxiliary,
        "detail": finding.detail,
    })
}

/// A report as JSON, with every evaluation and the keys and stages by label.
#[must_use]
pub fn staleness_report_json(report: &StalenessReport) -> Value {
    json!({
        "recommendation": report.recommendation.label(),
        "next_attempt": report.next_attempt,
        "evaluations": report.evaluations.iter().map(staleness_finding_json).collect::<Vec<_>>(),
        "not_evaluated": report
            .not_evaluated
            .iter()
            .map(|(what, why)| json!({"diagnostic": what, "reason": why}))
            .collect::<Vec<_>>(),
        "changed_keys": report.changed_key_labels(),
        "invalidated": report.invalidated_labels(),
        "changed_classes": report.changed_classes.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
        "invalidated_products": report
            .invalidated_products
            .iter()
            .map(|p| p.as_str())
            .collect::<Vec<_>>(),
        "plan_identity": report.plan_identity,
        "old_snapshot": report.old_snapshot,
        "new_snapshot": report.new_snapshot,
        "baseline_identity": report.baseline_identity,
        "policy_identity": report.policy_identity,
        "new_rows": report.new_rows,
        "identity": report.identity,
    })
}

fn field<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|value| !value.is_null())
}

fn real(object: &Map<String, Value>, key: &'static str, slot: &mut f64) -> Result<(), CausalError> {
    if let Some(value) = field(object, key) {
        *slot = value.as_f64().ok_or_else(|| CausalError::Compile {
            message: format!("staleness policy field {key} must be a number"),
        })?;
    }
    Ok(())
}

fn count(
    object: &Map<String, Value>,
    key: &'static str,
    slot: &mut u64,
) -> Result<(), CausalError> {
    if let Some(value) = field(object, key) {
        *slot = value.as_u64().ok_or_else(|| CausalError::Compile {
            message: format!("staleness policy field {key} must be a nonnegative integer"),
        })?;
    }
    Ok(())
}

/// A policy from a JSON object: every field is optional and defaults to
/// [`StalenessPolicy::default`]; an unknown field is refused.
///
/// # Errors
///
/// Malformed JSON, an unknown field, a field of the wrong type, or a policy that does not
/// validate.
pub fn staleness_policy_from_json(text: &str) -> Result<StalenessPolicy, CausalError> {
    const FIELDS: [&str; 25] = [
        "alpha",
        "multiplicity",
        "smd_threshold",
        "psi_threshold",
        "mahalanobis_threshold",
        "bins",
        "subsample",
        "permutations",
        "seed",
        "positivity_epsilon",
        "max_extreme_fraction",
        "min_ess_ratio_change",
        "balance_threshold",
        "max_condition",
        "min_complete_fraction",
        "support_tolerance",
        "min_level_rows",
        "missingness_threshold",
        "min_rows",
        "min_rows_per_arm",
        "min_rows_per_period",
        "period_column",
        "prior_variance",
        "prior_shape",
        "prior_rate",
    ];
    let parsed: Value = serde_json::from_str(text)
        .map_err(|error| CausalError::Compile { message: format!("staleness policy: {error}") })?;
    let Value::Object(object) = parsed else {
        return Err(CausalError::Compile { message: "staleness policy must be an object".into() });
    };
    if let Some(unknown) = object.keys().find(|key| !FIELDS.contains(&key.as_str())) {
        return Err(CausalError::Compile {
            message: format!("staleness policy has no field {unknown}"),
        });
    }
    let mut policy = StalenessPolicy::default();
    for (key, slot) in [
        ("alpha", &mut policy.alpha),
        ("smd_threshold", &mut policy.smd_threshold),
        ("psi_threshold", &mut policy.psi_threshold),
        ("mahalanobis_threshold", &mut policy.mahalanobis_threshold),
        ("positivity_epsilon", &mut policy.positivity_epsilon),
        ("max_extreme_fraction", &mut policy.max_extreme_fraction),
        ("min_ess_ratio_change", &mut policy.min_ess_ratio_change),
        ("balance_threshold", &mut policy.balance_threshold),
        ("max_condition", &mut policy.max_condition),
        ("min_complete_fraction", &mut policy.min_complete_fraction),
        ("support_tolerance", &mut policy.support_tolerance),
        ("missingness_threshold", &mut policy.missingness_threshold),
        ("prior_variance", &mut policy.prior_variance),
        ("prior_shape", &mut policy.prior_shape),
        ("prior_rate", &mut policy.prior_rate),
    ] {
        real(&object, key, slot)?;
    }
    for (key, slot) in [
        ("seed", &mut policy.seed),
        ("min_level_rows", &mut policy.min_level_rows),
        ("min_rows", &mut policy.min_rows),
        ("min_rows_per_arm", &mut policy.min_rows_per_arm),
        ("min_rows_per_period", &mut policy.min_rows_per_period),
    ] {
        count(&object, key, slot)?;
    }
    let mut wide = [policy.bins as u64, policy.subsample as u64, u64::from(policy.permutations)];
    for (key, slot) in ["bins", "subsample", "permutations"].into_iter().zip(wide.iter_mut()) {
        count(&object, key, slot)?;
    }
    let too_large = |key: &str| CausalError::Compile {
        message: format!("staleness policy field {key} is too large"),
    };
    policy.bins = usize::try_from(wide[0]).map_err(|_| too_large("bins"))?;
    policy.subsample = usize::try_from(wide[1]).map_err(|_| too_large("subsample"))?;
    policy.permutations = u32::try_from(wide[2]).map_err(|_| too_large("permutations"))?;
    if let Some(value) = field(&object, "multiplicity") {
        policy.multiplicity =
            value.as_str().and_then(Multiplicity::parse).ok_or_else(|| CausalError::Compile {
                message: "staleness policy multiplicity must be \"holm\" or \
                          \"benjamini_hochberg\""
                    .into(),
            })?;
    }
    if let Some(value) = field(&object, "period_column") {
        policy.period_column = Some(
            value
                .as_str()
                .ok_or_else(|| CausalError::Compile {
                    message: "staleness policy period_column must be a string".into(),
                })?
                .to_owned(),
        );
    }
    policy.validate().map_err(causal)?;
    Ok(policy)
}

/// The JSON object of a policy (every field), the inverse of [`staleness_policy_from_json`].
#[must_use]
pub fn staleness_policy_json(policy: &StalenessPolicy) -> Value {
    json!({
        "alpha": policy.alpha,
        "multiplicity": policy.multiplicity.label(),
        "smd_threshold": policy.smd_threshold,
        "psi_threshold": policy.psi_threshold,
        "mahalanobis_threshold": policy.mahalanobis_threshold,
        "bins": policy.bins,
        "subsample": policy.subsample,
        "permutations": policy.permutations,
        "seed": policy.seed,
        "positivity_epsilon": policy.positivity_epsilon,
        "max_extreme_fraction": policy.max_extreme_fraction,
        "min_ess_ratio_change": policy.min_ess_ratio_change,
        "balance_threshold": policy.balance_threshold,
        "max_condition": policy.max_condition,
        "min_complete_fraction": policy.min_complete_fraction,
        "support_tolerance": policy.support_tolerance,
        "min_level_rows": policy.min_level_rows,
        "missingness_threshold": policy.missingness_threshold,
        "min_rows": policy.min_rows,
        "min_rows_per_arm": policy.min_rows_per_arm,
        "min_rows_per_period": policy.min_rows_per_period,
        "period_column": policy.period_column,
        "prior_variance": policy.prior_variance,
        "prior_shape": policy.prior_shape,
        "prior_rate": policy.prior_rate,
        "identity": policy.identity(),
    })
}
