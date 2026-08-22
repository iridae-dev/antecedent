//! Analysis / estimate / identification / refutation / diagnostic wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, IdentificationStatus, VariableId,
};
use antecedent_estimate::{
    ClipSensitivity, EffectEstimate, FirstStageDiagnostics, OverlapPolicy, OverlapReport,
    PropensityInterval,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};
use antecedent_validate::RefutationReport;
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;
use crate::expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire};
use crate::query_wire::{CausalQueryWire, causal_query_from_wire, causal_query_to_wire};
use crate::trace::{
    AssumptionRecordWire, DerivationStepWire, assumptions_from_wire, assumptions_to_wire,
};

/// Effect estimate wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EffectEstimateWire {
    /// ATE.
    pub ate: f64,
    /// Analytic SE.
    pub se_analytic: f64,
    /// Bootstrap SE.
    pub se_bootstrap: Option<f64>,
    /// Bootstrap ok.
    pub bootstrap_replicates_ok: Option<u32>,
    /// Bootstrap failed.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Whether bootstrap stopped cooperatively after cancellation.
    #[serde(default)]
    pub bootstrap_cancelled: bool,
    /// Whether adaptive bootstrap stopped after convergence.
    #[serde(default)]
    pub bootstrap_early_stopped: bool,
    /// Assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Overlap policy tag.
    pub overlap_policy: String,
    /// Clip.
    pub overlap_clip: Option<f64>,
    /// Trim.
    pub overlap_trim: Option<f64>,
    /// Propensity-overlap evidence, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlap_report: Option<OverlapReportWire>,
    /// Weak-instrument first-stage evidence, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_stage_diagnostics: Option<FirstStageDiagnosticsWire>,
    /// Retained memory.
    pub retained_memory_bytes: Option<u64>,
}

/// Closed excluded propensity interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct PropensityIntervalWire {
    pub low: f64,
    pub high: f64,
}

/// Sensitivity of overlap evidence to clipping thresholds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct ClipSensitivityWire {
    pub thresholds: Vec<f64>,
    pub ess: Vec<f64>,
    pub treated_ess: Vec<f64>,
    pub control_ess: Vec<f64>,
    pub extreme_weight_counts: Vec<u32>,
}

/// Propensity overlap evidence retained on an effect estimate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct OverlapReportWire {
    pub propensity_min: f64,
    pub propensity_max: f64,
    pub ess: Option<f64>,
    pub extreme_weight_count: u32,
    pub excluded_fraction: f64,
    pub target_population_support: f64,
    pub excluded_regions: Vec<PropensityIntervalWire>,
    pub clip: Option<f64>,
    pub trim: Option<f64>,
    pub retained_fraction: f64,
    pub clip_sensitivity: Option<ClipSensitivityWire>,
}

/// Weak-instrument first-stage evidence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct FirstStageDiagnosticsWire {
    pub f_statistic: f64,
    pub df1: u64,
    pub df2: u64,
    pub partial_r2: f64,
}

/// Sharp RD design on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RdDesignWire {
    /// Running variable raw id.
    pub running_variable: u32,
    /// Cutoff.
    pub cutoff: f64,
    /// Bandwidth.
    pub bandwidth: f64,
}

/// Identified estimand wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentifiedEstimandWire {
    /// Method.
    pub method: String,
    /// Adjustment set.
    pub adjustment_set: Vec<u32>,
    /// Instruments.
    pub instruments: Vec<u32>,
    /// Mediators.
    pub mediators: Vec<u32>,
    /// Functional expr id.
    pub functional: u32,
    /// Optional sharp RD design.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rd_design: Option<RdDesignWire>,
}

/// Identification result wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentificationResultWire {
    /// Status.
    pub status: String,
    /// Query.
    pub query: CausalQueryWire,
    /// Estimands.
    pub estimands: Vec<IdentifiedEstimandWire>,
    /// Arena.
    pub arena: ExprArenaWire,
    /// Derivation.
    pub derivation: Vec<DerivationStepWire>,
    /// Assumptions.
    pub required_assumptions: Vec<AssumptionRecordWire>,
    /// Diagnostics.
    pub diagnostics: Vec<DiagnosticWire>,
    /// Performance.
    pub candidates_examined: u64,
    /// Sets returned.
    pub sets_returned: u64,
}

/// Diagnostic wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiagnosticWire {
    /// Code.
    pub code: String,
    /// Kind.
    pub kind: String,
    /// Severity.
    pub severity: String,
    /// Message.
    pub message: String,
    /// Artifact id.
    pub artifact_id: Option<String>,
    /// Structured diagnostic evidence retained losslessly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<(String, String)>,
}

/// Refutation report wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RefutationReportWire {
    /// Refuter.
    pub refuter: String,
    /// Original ATE.
    pub original_ate: f64,
    /// Refuted ATE.
    pub refuted_ate: f64,
    /// Comparison.
    pub comparison: f64,
    /// Informative.
    pub informative: bool,
    /// Passed.
    pub passed: bool,
    /// Failure condition.
    pub failure_condition: Option<String>,
    /// Replicates.
    pub replicates: u32,
}

/// Encode effect estimate.
#[must_use]
pub fn effect_estimate_to_wire(e: &EffectEstimate) -> EffectEstimateWire {
    let (overlap_policy, overlap_clip, overlap_trim) = match e.overlap {
        OverlapPolicy::ExplicitOverride => ("explicit_override".into(), None, None),
        OverlapPolicy::RequireDiagnostics { clip, trim } => {
            ("require_diagnostics".into(), clip, trim)
        }
    };
    EffectEstimateWire {
        ate: e.ate,
        se_analytic: e.se_analytic,
        se_bootstrap: e.se_bootstrap,
        bootstrap_replicates_ok: e.bootstrap_replicates_ok,
        bootstrap_replicates_failed: e.bootstrap_replicates_failed,
        bootstrap_cancelled: e.bootstrap_cancelled,
        bootstrap_early_stopped: e.bootstrap_early_stopped,
        assumptions: assumptions_to_wire(&e.assumptions),
        overlap_policy,
        overlap_clip,
        overlap_trim,
        overlap_report: e.overlap_report.as_ref().map(overlap_report_to_wire),
        first_stage_diagnostics: e.first_stage_diagnostics.as_ref().map(|diagnostic| {
            FirstStageDiagnosticsWire {
                f_statistic: diagnostic.f_statistic,
                df1: u64::try_from(diagnostic.df1).unwrap_or(u64::MAX),
                df2: u64::try_from(diagnostic.df2).unwrap_or(u64::MAX),
                partial_r2: diagnostic.partial_r2,
            }
        }),
        retained_memory_bytes: e.retained_memory_bytes,
    }
}

fn overlap_report_to_wire(report: &OverlapReport) -> OverlapReportWire {
    OverlapReportWire {
        propensity_min: report.propensity_min,
        propensity_max: report.propensity_max,
        ess: report.ess,
        extreme_weight_count: report.extreme_weight_count,
        excluded_fraction: report.excluded_fraction,
        target_population_support: report.target_population_support,
        excluded_regions: report
            .excluded_regions
            .iter()
            .map(|region| PropensityIntervalWire { low: region.low, high: region.high })
            .collect(),
        clip: report.clip,
        trim: report.trim,
        retained_fraction: report.retained_fraction,
        clip_sensitivity: report.clip_sensitivity.as_ref().map(|sensitivity| ClipSensitivityWire {
            thresholds: sensitivity.thresholds.to_vec(),
            ess: sensitivity.ess.to_vec(),
            treated_ess: sensitivity.treated_ess.to_vec(),
            control_ess: sensitivity.control_ess.to_vec(),
            extreme_weight_counts: sensitivity.extreme_weight_counts.to_vec(),
        }),
    }
}

/// Decode an effect estimate without dropping its assumptions or diagnostic evidence.
///
/// # Errors
///
/// Invalid assumption labels or scopes.
pub fn effect_estimate_from_wire(w: &EffectEstimateWire) -> Result<EffectEstimate, IoError> {
    let overlap = match w.overlap_policy.as_str() {
        "require_diagnostics" => {
            OverlapPolicy::RequireDiagnostics { clip: w.overlap_clip, trim: w.overlap_trim }
        }
        "explicit_override" => OverlapPolicy::ExplicitOverride,
        other => return Err(IoError::Convert(format!("unknown overlap policy `{other}`"))),
    };
    let overlap_report = w.overlap_report.as_ref().map(overlap_report_from_wire).transpose()?;
    let first_stage = w
        .first_stage_diagnostics
        .as_ref()
        .map(|diagnostic| {
            if !diagnostic.f_statistic.is_finite()
                || diagnostic.f_statistic < 0.0
                || !diagnostic.partial_r2.is_finite()
                || !(0.0..=1.0).contains(&diagnostic.partial_r2)
                || diagnostic.df1 == 0
                || diagnostic.df2 == 0
            {
                return Err(IoError::Convert("invalid first-stage diagnostic evidence".into()));
            }
            Ok(FirstStageDiagnostics {
                f_statistic: diagnostic.f_statistic,
                df1: usize::try_from(diagnostic.df1).map_err(|_| IoError::TooLarge)?,
                df2: usize::try_from(diagnostic.df2).map_err(|_| IoError::TooLarge)?,
                partial_r2: diagnostic.partial_r2,
            })
        })
        .transpose()?;
    Ok(EffectEstimate::from_parts(
        w.ate,
        w.se_analytic,
        w.se_bootstrap,
        w.bootstrap_replicates_ok,
        w.bootstrap_replicates_failed,
        w.bootstrap_cancelled,
        w.bootstrap_early_stopped,
        assumptions_from_wire(&w.assumptions)?,
        overlap,
        overlap_report,
        w.retained_memory_bytes,
    )
    .with_first_stage_diagnostics(first_stage))
}

fn overlap_report_from_wire(wire: &OverlapReportWire) -> Result<OverlapReport, IoError> {
    let finite = [
        wire.propensity_min,
        wire.propensity_max,
        wire.excluded_fraction,
        wire.target_population_support,
        wire.retained_fraction,
    ]
    .into_iter()
    .chain(wire.ess)
    .all(f64::is_finite);
    if !finite
        || wire.propensity_min < 0.0
        || wire.propensity_min > wire.propensity_max
        || wire.propensity_max > 1.0
        || wire.ess.is_some_and(|value| value < 0.0)
        || !(0.0..=1.0).contains(&wire.excluded_fraction)
        || !(0.0..=1.0).contains(&wire.target_population_support)
        || !(0.0..=1.0).contains(&wire.retained_fraction)
        || wire.excluded_regions.iter().any(|region| {
            !region.low.is_finite()
                || !region.high.is_finite()
                || region.low < 0.0
                || region.low > region.high
                || region.high > 1.0
        })
    {
        return Err(IoError::Convert("invalid propensity-overlap evidence".into()));
    }
    if let Some(sensitivity) = &wire.clip_sensitivity {
        let len = sensitivity.thresholds.len();
        if len == 0
            || sensitivity.ess.len() != len
            || sensitivity.treated_ess.len() != len
            || sensitivity.control_ess.len() != len
            || sensitivity.extreme_weight_counts.len() != len
            || sensitivity
                .thresholds
                .iter()
                .chain(&sensitivity.ess)
                .chain(&sensitivity.treated_ess)
                .chain(&sensitivity.control_ess)
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(IoError::Convert("invalid overlap clip-sensitivity evidence".into()));
        }
    }
    Ok(OverlapReport {
        propensity_min: wire.propensity_min,
        propensity_max: wire.propensity_max,
        ess: wire.ess,
        extreme_weight_count: wire.extreme_weight_count,
        excluded_fraction: wire.excluded_fraction,
        target_population_support: wire.target_population_support,
        excluded_regions: wire
            .excluded_regions
            .iter()
            .map(|region| PropensityInterval { low: region.low, high: region.high })
            .collect::<Vec<_>>()
            .into(),
        clip: wire.clip,
        trim: wire.trim,
        retained_fraction: wire.retained_fraction,
        clip_sensitivity: wire.clip_sensitivity.as_ref().map(|sensitivity| ClipSensitivity {
            thresholds: sensitivity.thresholds.clone().into(),
            ess: sensitivity.ess.clone().into(),
            treated_ess: sensitivity.treated_ess.clone().into(),
            control_ess: sensitivity.control_ess.clone().into(),
            extreme_weight_counts: sensitivity.extreme_weight_counts.clone().into(),
        }),
    })
}

/// Encode identification result.
///
/// # Errors
///
/// Query encode failures.
pub fn identification_to_wire(
    r: &IdentificationResult,
) -> Result<IdentificationResultWire, IoError> {
    Ok(IdentificationResultWire {
        status: match r.status {
            IdentificationStatus::NonparametricallyIdentified => {
                "nonparametrically_identified".into()
            }
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                "identified_under_parametric_restrictions".into()
            }
            IdentificationStatus::IdentifiedUnderPriorRestrictions => {
                "identified_under_prior_restrictions".into()
            }
            IdentificationStatus::PartiallyIdentified => "partially_identified".into(),
            IdentificationStatus::GraphDependent => "graph_dependent".into(),
            IdentificationStatus::NotIdentified => "not_identified".into(),
        },
        query: causal_query_to_wire(&r.query)?,
        estimands: r
            .estimands
            .iter()
            .map(|e| IdentifiedEstimandWire {
                method: e.method.to_string(),
                adjustment_set: vars_to_raw(&e.adjustment_set),
                instruments: vars_to_raw(&e.instruments),
                mediators: vars_to_raw(&e.mediators),
                functional: e.functional.raw(),
                rd_design: e.rd_design.map(|d| RdDesignWire {
                    running_variable: d.running_variable.raw(),
                    cutoff: d.cutoff,
                    bandwidth: d.bandwidth,
                }),
            })
            .collect(),
        arena: expr_arena_to_wire(&r.arena)?,
        derivation: r
            .derivation
            .steps
            .iter()
            .map(|s| DerivationStepWire { rule: s.rule.to_string(), detail: s.detail.to_string() })
            .collect(),
        required_assumptions: assumptions_to_wire(&r.required_assumptions),
        diagnostics: r.diagnostics.iter().map(diagnostic_to_wire).collect(),
        candidates_examined: r.performance.candidates_examined,
        sets_returned: r.performance.sets_returned,
    })
}

/// Decode identification result.
///
/// # Errors
///
/// Unknown status / query / arena.
pub fn identification_from_wire(
    w: &IdentificationResultWire,
) -> Result<IdentificationResult, IoError> {
    let status = match w.status.as_str() {
        "nonparametrically_identified" => IdentificationStatus::NonparametricallyIdentified,
        "identified_under_parametric_restrictions" => {
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        }
        "identified_under_prior_restrictions" => {
            IdentificationStatus::IdentifiedUnderPriorRestrictions
        }
        "partially_identified" => IdentificationStatus::PartiallyIdentified,
        "graph_dependent" => IdentificationStatus::GraphDependent,
        "not_identified" => IdentificationStatus::NotIdentified,
        other => {
            return Err(IoError::Convert(format!("unknown IdentificationStatus `{other}`")));
        }
    };
    Ok(IdentificationResult::from_parts(
        status,
        causal_query_from_wire(&w.query)?,
        w.estimands
            .iter()
            .map(|e| {
                IdentifiedEstimand::new(
                    Arc::from(e.method.as_str()),
                    vars_from_raw(&e.adjustment_set),
                    vars_from_raw(&e.instruments),
                    vars_from_raw(&e.mediators),
                    ExprId::from_raw(e.functional),
                    e.rd_design.as_ref().map(|d| {
                        antecedent_expr::RdDesignParams::new(
                            VariableId::from_raw(d.running_variable),
                            d.cutoff,
                            d.bandwidth,
                        )
                    }),
                )
            })
            .collect(),
        expr_arena_from_wire(&w.arena)?,
        DerivationTrace {
            steps: w
                .derivation
                .iter()
                .map(|s| antecedent_identify::DerivationStep {
                    rule: Arc::from(s.rule.as_str()),
                    detail: Arc::from(s.detail.as_str()),
                })
                .collect(),
        },
        assumptions_from_wire(&w.required_assumptions)?,
        w.diagnostics.iter().map(diagnostic_from_wire).collect::<Result<Vec<_>, _>>()?,
        IdentificationPerformanceRecord {
            candidates_examined: w.candidates_examined,
            sets_returned: w.sets_returned,
        },
        None,
    ))
}

/// Encode refutation.
#[must_use]
pub fn refutation_to_wire(r: &RefutationReport) -> RefutationReportWire {
    RefutationReportWire {
        refuter: r.refuter.to_string(),
        original_ate: r.original_ate,
        refuted_ate: r.refuted_ate,
        comparison: r.comparison,
        informative: r.informative,
        passed: r.passed,
        failure_condition: r.failure_condition.as_ref().map(ToString::to_string),
        replicates: r.replicates,
    }
}

/// Decode refutation.
#[must_use]
pub fn refutation_from_wire(w: &RefutationReportWire) -> RefutationReport {
    RefutationReport::new(
        Arc::from(w.refuter.as_str()),
        w.original_ate,
        w.refuted_ate,
        w.comparison,
        w.informative,
        w.passed,
        w.failure_condition.as_ref().map(|s| Arc::<str>::from(s.as_str())),
        w.replicates,
    )
}

/// Encode diagnostic.
#[must_use]
pub fn diagnostic_to_wire(d: &Diagnostic) -> DiagnosticWire {
    DiagnosticWire {
        code: d.code.to_string(),
        kind: match d.kind {
            DiagnosticKind::Scientific => "scientific",
            DiagnosticKind::Execution => "execution",
        }
        .into(),
        severity: match d.severity {
            DiagnosticSeverity::Info => "info",
            DiagnosticSeverity::Warning => "warning",
            DiagnosticSeverity::Error => "error",
        }
        .into(),
        message: d.message.to_string(),
        artifact_id: d.artifact_id.as_ref().map(ToString::to_string),
        fields: d.fields.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
    }
}

/// Decode diagnostic.
///
/// # Errors
///
/// Unknown kind/severity.
pub fn diagnostic_from_wire(w: &DiagnosticWire) -> Result<Diagnostic, IoError> {
    Ok(Diagnostic {
        code: Arc::from(w.code.as_str()),
        kind: match w.kind.as_str() {
            "scientific" => DiagnosticKind::Scientific,
            "execution" => DiagnosticKind::Execution,
            other => return Err(IoError::Convert(format!("unknown DiagnosticKind `{other}`"))),
        },
        severity: match w.severity.as_str() {
            "info" => DiagnosticSeverity::Info,
            "warning" => DiagnosticSeverity::Warning,
            "error" => DiagnosticSeverity::Error,
            other => {
                return Err(IoError::Convert(format!("unknown DiagnosticSeverity `{other}`")));
            }
        },
        message: Arc::from(w.message.as_str()),
        artifact_id: w.artifact_id.as_ref().map(|a| Arc::<str>::from(a.as_str())),
        fields: w
            .fields
            .iter()
            .map(|(key, value)| (Arc::from(key.as_str()), Arc::from(value.as_str())))
            .collect::<Vec<_>>()
            .into(),
    })
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery};
    use antecedent_expr::CausalExprArena;

    use super::*;
    use crate::trace::{AssumptionRecordWire, AssumptionTagWire};

    fn empty_id_result(status: IdentificationStatus) -> IdentificationResult {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        IdentificationResult::from_parts(
            status,
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(t, y)),
            Vec::new(),
            CausalExprArena::new(),
            DerivationTrace::default(),
            AssumptionSet::default(),
            Vec::new(),
            IdentificationPerformanceRecord::default(),
            None,
        )
    }

    #[test]
    fn restricted_status_wire_round_trips() {
        for status in [
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            IdentificationStatus::IdentifiedUnderPriorRestrictions,
        ] {
            let wire = identification_to_wire(&empty_id_result(status)).unwrap();
            let back = identification_from_wire(&wire).unwrap();
            assert_eq!(back.status, status);
        }
    }

    fn descriptive_assumption() -> AssumptionRecordWire {
        AssumptionRecordWire {
            assumption: AssumptionTagWire::Custom {
                id: "stable.assumption.id".into(),
                description: Some("scientifically meaningful description".into()),
            },
            source: "derived:diagnostic-7".into(),
            scope: "variables:[0,1]".into(),
            status: "contradicted".into(),
        }
    }

    #[test]
    fn effect_estimate_preserves_assumption_evidence() {
        let wire = EffectEstimateWire {
            ate: 1.0,
            se_analytic: 0.1,
            se_bootstrap: Some(0.2),
            bootstrap_replicates_ok: Some(99),
            bootstrap_replicates_failed: Some(1),
            bootstrap_cancelled: true,
            bootstrap_early_stopped: true,
            assumptions: vec![descriptive_assumption()],
            overlap_policy: "require_diagnostics".into(),
            overlap_clip: Some(0.01),
            overlap_trim: Some(0.02),
            overlap_report: Some(OverlapReportWire {
                propensity_min: 0.01,
                propensity_max: 0.98,
                ess: Some(42.0),
                extreme_weight_count: 2,
                excluded_fraction: 0.1,
                target_population_support: 0.9,
                excluded_regions: vec![PropensityIntervalWire { low: 0.0, high: 0.02 }],
                clip: Some(0.01),
                trim: Some(0.02),
                retained_fraction: 0.9,
                clip_sensitivity: Some(ClipSensitivityWire {
                    thresholds: vec![0.005, 0.01],
                    ess: vec![40.0, 42.0],
                    treated_ess: vec![20.0, 21.0],
                    control_ess: vec![20.0, 21.0],
                    extreme_weight_counts: vec![3, 2],
                }),
            }),
            first_stage_diagnostics: Some(FirstStageDiagnosticsWire {
                f_statistic: 12.0,
                df1: 1,
                df2: 98,
                partial_r2: 0.2,
            }),
            retained_memory_bytes: Some(4096),
        };
        let domain = effect_estimate_from_wire(&wire).unwrap();
        assert_eq!(effect_estimate_to_wire(&domain), wire);
    }

    #[test]
    fn identification_result_preserves_required_assumption_evidence() {
        let mut wire = identification_to_wire(&empty_id_result(
            IdentificationStatus::NonparametricallyIdentified,
        ))
        .unwrap();
        wire.required_assumptions = vec![descriptive_assumption()];
        let domain = identification_from_wire(&wire).unwrap();
        assert_eq!(
            identification_to_wire(&domain).unwrap().required_assumptions,
            wire.required_assumptions
        );
    }

    #[test]
    fn diagnostic_preserves_structured_evidence_fields() {
        let diagnostic = Diagnostic {
            code: Arc::from("support.overlap"),
            kind: DiagnosticKind::Scientific,
            severity: DiagnosticSeverity::Warning,
            message: Arc::from("weak overlap"),
            artifact_id: Some(Arc::from("artifact-7")),
            fields: Arc::from([
                (Arc::from("minimum_probability"), Arc::from("0.01")),
                (Arc::from("effective_sample_size"), Arc::from("12.5")),
            ]),
        };
        let wire = diagnostic_to_wire(&diagnostic);
        assert_eq!(diagnostic_from_wire(&wire).unwrap(), diagnostic);
    }

    #[test]
    fn distribution_and_path_specific_query_identification_wire() {
        use antecedent_core::{
            Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, Value,
        };

        let dist_q = CausalQuery::Distribution(
            InterventionalDistributionQuery::new(
                VariableId::from_raw(1),
                [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
            )
            .with_conditioning([VariableId::from_raw(2)]),
        );
        let mut dist = empty_id_result(IdentificationStatus::NonparametricallyIdentified);
        dist.query = dist_q;
        let wire = identification_to_wire(&dist).unwrap();
        let back = identification_from_wire(&wire).unwrap();
        assert!(matches!(
            back.query,
            CausalQuery::Distribution(q) if q.conditioning.len() == 1
        ));

        let path_q = CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
                .with_path_nodes([VariableId::from_raw(1)]),
        );
        let mut path = empty_id_result(IdentificationStatus::NonparametricallyIdentified);
        path.query = path_q;
        path.estimands.push(IdentifiedEstimand::new(
            Arc::from("path_specific.natural"),
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
            ExprId::from_raw(0),
            None,
        ));
        let wire = identification_to_wire(&path).unwrap();
        assert_eq!(wire.estimands[0].method, "path_specific.natural");
        let back = identification_from_wire(&wire).unwrap();
        assert!(matches!(
            back.query,
            CausalQuery::PathSpecific(q) if q.path_nodes.len() == 1
        ));
        assert_eq!(back.estimands[0].method.as_ref(), "path_specific.natural");
    }
}
