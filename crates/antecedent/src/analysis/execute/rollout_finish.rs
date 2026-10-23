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
    let mut notes = report.notes;
    notes.push(Diagnostic::new(
        "estimate.quasi.rollout.unlicensed",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "unlicensed route: the {} intervals come from the design's whole-procedure path \
             ({}); their repeated-sampling coverage is unmeasured and no scalar interval is \
             published on the effect",
            report.design, report.uncertainty
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
    result.payload =
        crate::result::ResultPayload::QuasiRollout(Box::new(crate::QuasiRolloutEstimate {
            design: Arc::from(report.design),
            method: Arc::from(report.method),
            effects: Arc::from(report.effects),
            uncertainty: Arc::from(report.uncertainty),
            calibration: Arc::from(UNMEASURED),
            replicates_requested: report.replicates[0],
            replicates_ok: report.replicates[1],
            replicates_failed: report.replicates[2],
            critical_value: report.critical_value,
            diagnostics: report
                .diagnostics
                .into_iter()
                .map(|(name, value)| (Arc::<str>::from(name.as_str()), value))
                .collect(),
            licensed: false,
        }));
    result.treatment = None;
    result.support_status = None;
    result
}
