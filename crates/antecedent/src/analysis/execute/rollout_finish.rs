//! Result assembly shared by the rollout designs: heterogeneity-robust and covariate-adjusted
//! difference in differences, conformal and multi-unit synthetic control, and the local ratio
//! with a selected bandwidth or a calendar-time running variable.
//!
//! Every rollout result is unlicensed: its scalar summary carries no published interval, the
//! design's own whole-procedure intervals ride on the [`crate::QuasiRolloutEstimate`] payload,
//! and their calibration is `unmeasured` until a coverage record measures them.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId,
    IdentifiedExecuteFinish, IdentifierId, RandomizedTrialOperation, RandomizedTrialQuery,
    Strategy, StudyResult, TabularData, finish_identified_execute_with_context,
};
use antecedent_core::VariableId;
use antecedent_estimate::rollout_resampling::DesignEffect;

/// Calibration label of every rollout interval.
pub(super) const UNMEASURED: &str = "unmeasured";

/// Named diagnostics from fixed names.
pub(super) fn named(values: &[(&str, f64)]) -> Vec<(String, f64)> {
    values.iter().map(|(name, value)| ((*name).to_string(), *value)).collect()
}

/// What one rollout execution reports.
pub(super) struct RolloutReport {
    /// Design name of the payload.
    pub(super) design: &'static str,
    /// Estimator, comparison, adjustment and aggregation words.
    pub(super) method: String,
    /// Reported effects; the first is the scalar summary.
    pub(super) effects: Vec<DesignEffect>,
    /// How the uncertainty was computed.
    pub(super) uncertainty: &'static str,
    /// Requested, successful and failed replicates.
    pub(super) replicates: [u32; 3],
    /// Sup-t critical value of the simultaneous band.
    pub(super) critical_value: Option<f64>,
    /// Named numeric diagnostics.
    pub(super) diagnostics: Vec<(String, f64)>,
    /// Scientific notes beside the result.
    pub(super) notes: Vec<Diagnostic>,
}

/// The payload fields of a rollout report, without its notes.
pub(super) struct RolloutPayloadParts {
    pub(super) design: &'static str,
    pub(super) method: String,
    pub(super) effects: Vec<DesignEffect>,
    pub(super) uncertainty: &'static str,
    pub(super) replicates: [u32; 3],
    pub(super) critical_value: Option<f64>,
    pub(super) diagnostics: Vec<(String, f64)>,
}

/// The unlicensed rollout payload of `parts`; calibration is always `unmeasured`.
pub(super) fn rollout_payload(parts: RolloutPayloadParts) -> crate::result::ResultPayload {
    crate::result::ResultPayload::QuasiRollout(Box::new(crate::QuasiRolloutEstimate {
        design: Arc::from(parts.design),
        method: Arc::from(parts.method),
        effects: Arc::from(parts.effects),
        uncertainty: Arc::from(parts.uncertainty),
        calibration: Arc::from(UNMEASURED),
        replicates_requested: parts.replicates[0],
        replicates_ok: parts.replicates[1],
        replicates_failed: parts.replicates[2],
        critical_value: parts.critical_value,
        diagnostics: parts
            .diagnostics
            .into_iter()
            .map(|(name, value)| (Arc::<str>::from(name.as_str()), value))
            .collect(),
        licensed: false,
    }))
}

/// Split a report into its payload parts and its notes.
pub(super) fn split_report(report: RolloutReport) -> (RolloutPayloadParts, Vec<Diagnostic>) {
    let RolloutReport {
        design,
        method,
        effects,
        uncertainty,
        replicates,
        critical_value,
        diagnostics,
        notes,
    } = report;
    (
        RolloutPayloadParts {
            design,
            method,
            effects,
            uncertainty,
            replicates,
            critical_value,
            diagnostics,
        },
        notes,
    )
}

/// The unlicensed result of a rollout execution under `operation`.
pub(super) fn finish_rollout<Q: Clone + RandomizedTrialQuery, E>(
    operation: &RandomizedTrialOperation<Q, E>,
    data: &TabularData,
    outcome: VariableId,
    report: RolloutReport,
) -> StudyResult {
    let overall = report.effects.first().map_or(f64::NAN, |effect| effect.estimate);
    let estimate = EffectEstimate::new(
        overall,
        f64::NAN,
        operation.identification.required_assumptions.clone(),
        antecedent_estimate::OverlapPolicy::ExplicitOverride,
    );
    let RolloutReport {
        design,
        method,
        effects,
        uncertainty,
        replicates,
        critical_value,
        diagnostics,
        notes,
    } = report;
    let mut notes = notes;
    let report_fields = RolloutPayloadParts {
        design,
        method,
        effects,
        uncertainty,
        replicates,
        critical_value,
        diagnostics,
    };
    notes.push(Diagnostic::new(
        "estimate.quasi.rollout.unlicensed",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "unlicensed route: the {} intervals come from the design's whole-procedure path \
             ({}); their repeated-sampling coverage is unmeasured and no scalar interval is \
             published on the effect",
            design, uncertainty
        ),
    ));
    let mut result = finish_identified_execute_with_context(
        &operation.result_context,
        Some(data),
        IdentifiedExecuteFinish::new(
            &operation.physical,
            operation.identification.clone(),
            operation.estimand.clone(),
            estimate,
            Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt),
            outcome,
            outcome,
        )
        .extra_diagnostics(notes),
    );
    result.payload = rollout_payload(report_fields);
    result.treatment = None;
    result.support_status = None;
    result
}
