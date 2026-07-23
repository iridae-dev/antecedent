//! Analysis / estimate / identification / refutation / diagnostic wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, IdentificationStatus, VariableId,
};
use causal_estimate::{EffectEstimate, OverlapPolicy};
use causal_expr::{ExprId, IdentifiedEstimand};
use causal_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};
use causal_validate::RefutationReport;
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;
use crate::expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire};
use crate::query_wire::{CausalQueryWire, causal_query_from_wire, causal_query_to_wire};
use crate::trace::{AssumptionRecordWire, DerivationStepWire, assumptions_to_wire};

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
    /// Assumptions.
    pub assumptions: Vec<AssumptionRecordWire>,
    /// Overlap policy tag.
    pub overlap_policy: String,
    /// Clip.
    pub overlap_clip: Option<f64>,
    /// Trim.
    pub overlap_trim: Option<f64>,
    /// Retained memory.
    pub retained_memory_bytes: Option<u64>,
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
        assumptions: assumptions_to_wire(&e.assumptions),
        overlap_policy,
        overlap_clip,
        overlap_trim,
        retained_memory_bytes: e.retained_memory_bytes,
    }
}

/// Decode effect estimate (overlap report dropped).
#[must_use]
pub fn effect_estimate_from_wire(w: &EffectEstimateWire) -> EffectEstimate {
    let overlap = match w.overlap_policy.as_str() {
        "require_diagnostics" => {
            OverlapPolicy::RequireDiagnostics { clip: w.overlap_clip, trim: w.overlap_trim }
        }
        _ => OverlapPolicy::ExplicitOverride,
    };
    EffectEstimate::from_parts(
        w.ate,
        w.se_analytic,
        w.se_bootstrap,
        w.bootstrap_replicates_ok,
        w.bootstrap_replicates_failed,
        false,
        false,
        causal_core::AssumptionSet::new(),
        overlap,
        None,
        w.retained_memory_bytes,
    )
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
                        causal_expr::RdDesignParams::new(
                            causal_core::VariableId::from_raw(d.running_variable),
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
                .map(|s| causal_identify::DerivationStep {
                    rule: Arc::from(s.rule.as_str()),
                    detail: Arc::from(s.detail.as_str()),
                })
                .collect(),
        },
        causal_core::AssumptionSet::new(),
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
        fields: Arc::from([]),
    })
}

/// Silence unused.
#[allow(dead_code)]
fn _keep(_: VariableId) {}

#[cfg(test)]
mod tests {
    use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery};
    use causal_expr::CausalExprArena;

    use super::*;

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

    #[test]
    fn distribution_and_path_specific_query_identification_wire() {
        use causal_core::{
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
