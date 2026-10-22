//! Identification, hedge and diagnostic wire types and their conversions.
//!
//! The effect-estimate and refutation wires, whose conversions need the
//! estimators, live in `antecedent-io` and join this module's public path
//! there.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, IdentificationStatus, VariableId,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_graph::DenseNodeId;
use antecedent_identify::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationResultParts,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;
use crate::expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire};
use crate::query_wire::{CausalQueryWire, causal_query_from_wire};
use crate::trace::{
    AssumptionRecordWire, DerivationStepWire, assumptions_from_wire, assumptions_to_wire,
};

/// Sharp RD design on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// ID-algorithm hedge witnessing that `not_identified` is a proof rather than a
    /// failure to find an identifier. Absent when no witness was produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hedge: Option<HedgeCertificateWire>,
}

/// Shpitser–Pearl hedge `(F, F')` on the wire: the two C-forests as variable ids and
/// dense graph coordinates, each strictly increasing.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HedgeCertificateWire {
    /// Variables of the larger forest `F`.
    pub f: Vec<u32>,
    /// Variables of the smaller forest `F'`.
    pub f_prime: Vec<u32>,
    /// Dense ids of `F`.
    pub f_dense: Vec<u32>,
    /// Dense ids of `F'`.
    pub f_prime_dense: Vec<u32>,
    /// The graph and query the hedge was found in; with it the witness re-verifies on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<HedgeProblemWire>,
}

/// ADMG plus treatments and outcomes a hedge witnesses (see `HedgeProblem`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HedgeProblemWire {
    /// Variable of each dense node, in dense order.
    pub variables: Vec<u32>,
    /// Directed edges by dense id.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges by dense id.
    pub bidirected: Vec<(u32, u32)>,
    /// Treatment variables.
    pub treatments: Vec<u32>,
    /// Outcome variables.
    pub outcomes: Vec<u32>,
}

impl HedgeCertificateWire {
    /// Every variable named by the witness.
    pub(crate) fn variables(&self) -> impl Iterator<Item = u32> + '_ {
        let problem = self
            .problem
            .iter()
            .flat_map(|p| p.variables.iter().chain(&p.treatments).chain(&p.outcomes).copied());
        self.f.iter().chain(&self.f_prime).copied().chain(problem)
    }

    /// Structural conditions of a hedge that need no graph: strictly increasing dense
    /// coordinates paired one-to-one with distinct variables, and `F'` a non-empty
    /// subset of `F` coordinate by coordinate. The graph conditions are re-verified by
    /// `HedgeCertificate::verify` wherever the graph is held.
    fn validate(&self) -> Result<(), IoError> {
        let invalid = |what: &str| IoError::Convert(format!("invalid hedge certificate: {what}"));
        if self.f.len() != self.f_dense.len() || self.f_prime.len() != self.f_prime_dense.len() {
            return Err(invalid("variable and dense id lists differ in length"));
        }
        let distinct = |ids: &[u32]| ids.iter().collect::<std::collections::BTreeSet<_>>().len();
        if [&self.f_dense, &self.f_prime_dense].iter().any(|d| d.windows(2).any(|w| w[0] >= w[1]))
            || distinct(&self.f) != self.f.len()
            || distinct(&self.f_prime) != self.f_prime.len()
        {
            return Err(invalid("dense ids must be strictly increasing and variables distinct"));
        }
        let forest: std::collections::BTreeMap<u32, u32> =
            self.f_dense.iter().copied().zip(self.f.iter().copied()).collect();
        let nested = self
            .f_prime_dense
            .iter()
            .zip(&self.f_prime)
            .all(|(dense, variable)| forest.get(dense) == Some(variable));
        if self.f_prime.is_empty() || !nested {
            return Err(invalid("F' must be a non-empty subset of F"));
        }
        Ok(())
    }
}

/// Diagnostic wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

/// The one table of identification-status spellings: `snake_case` on the analysis and
/// response wires, `PascalCase` on posterior artifacts and prior-bank metadata (the
/// host-facing label those durable formats and their consumers already carry).
pub(crate) const STATUS_SPELLINGS: [(IdentificationStatus, &str, &str); 6] = [
    (
        IdentificationStatus::NonparametricallyIdentified,
        "nonparametrically_identified",
        "NonparametricallyIdentified",
    ),
    (
        IdentificationStatus::IdentifiedUnderParametricRestrictions,
        "identified_under_parametric_restrictions",
        "IdentifiedUnderParametricRestrictions",
    ),
    (
        IdentificationStatus::IdentifiedUnderPriorRestrictions,
        "identified_under_prior_restrictions",
        "IdentifiedUnderPriorRestrictions",
    ),
    (IdentificationStatus::PartiallyIdentified, "partially_identified", "PartiallyIdentified"),
    (IdentificationStatus::GraphDependent, "graph_dependent", "GraphDependent"),
    (IdentificationStatus::NotIdentified, "not_identified", "NotIdentified"),
];

/// `snake_case` spelling of a status.
#[must_use]
pub(crate) fn identification_status_snake(status: IdentificationStatus) -> &'static str {
    STATUS_SPELLINGS.iter().find(|(s, _, _)| *s == status).map_or("not_identified", |row| row.1)
}

/// `PascalCase` spelling of a status.
// doc(hidden): shared with antecedent-io posterior conversion.
#[doc(hidden)]
#[must_use]
pub fn identification_status_pascal(status: IdentificationStatus) -> &'static str {
    STATUS_SPELLINGS.iter().find(|(s, _, _)| *s == status).map_or("NotIdentified", |row| row.2)
}

/// Status named by either spelling.
#[must_use]
pub(crate) fn identification_status_from_any(spelling: &str) -> Option<IdentificationStatus> {
    STATUS_SPELLINGS
        .iter()
        .find(|(_, snake, pascal)| *snake == spelling || *pascal == spelling)
        .map(|(status, _, _)| *status)
}

/// Encode identification result.
///
/// # Errors
///
/// Query encode failures.
pub fn identification_to_wire(
    r: &IdentificationResult,
) -> Result<IdentificationResultWire, IoError> {
    identification_to_wire_with_registry(r, None)
}

/// Encode an identification result whose query names registry populations.
///
/// # Errors
///
/// Query encode failure, or a named population missing from `registry`.
pub fn identification_to_wire_with_registry(
    r: &IdentificationResult,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> Result<IdentificationResultWire, IoError> {
    Ok(IdentificationResultWire {
        status: identification_status_snake(r.status).into(),
        query: crate::query_wire::causal_query_to_wire_with_registry(&r.query, registry)?,
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
        hedge: r.hedge.as_ref().map(|h| HedgeCertificateWire {
            f: vars_to_raw(&h.f),
            f_prime: vars_to_raw(&h.f_prime),
            f_dense: h.f_dense.iter().map(|d| d.raw()).collect(),
            f_prime_dense: h.f_prime_dense.iter().map(|d| d.raw()).collect(),
            problem: h.problem.as_ref().map(|p| HedgeProblemWire {
                variables: vars_to_raw(&p.variables),
                directed: p.directed.to_vec(),
                bidirected: p.bidirected.to_vec(),
                treatments: vars_to_raw(&p.treatments),
                outcomes: vars_to_raw(&p.outcomes),
            }),
        }),
    })
}

/// Decode identification result.
///
/// # Errors
///
/// Unknown status / query / arena.
#[allow(
    clippy::too_many_lines,
    reason = "one conversion per wire field of the identification result, in declaration order"
)]
pub fn identification_from_wire(
    w: &IdentificationResultWire,
) -> Result<IdentificationResult, IoError> {
    let status = STATUS_SPELLINGS
        .iter()
        .find(|(_, snake, _)| *snake == w.status)
        .map(|(status, _, _)| *status)
        .ok_or_else(|| IoError::Convert(format!("unknown IdentificationStatus `{}`", w.status)))?;
    let hedge = match &w.hedge {
        Some(wire) => {
            if status != IdentificationStatus::NotIdentified {
                return Err(IoError::Convert(
                    "a hedge witness is only meaningful for a not_identified result".into(),
                ));
            }
            wire.validate()?;
            let certificate = antecedent_identify::HedgeCertificate {
                f: vars_from_raw(&wire.f),
                f_prime: vars_from_raw(&wire.f_prime),
                f_dense: wire.f_dense.iter().copied().map(DenseNodeId::from_raw).collect(),
                f_prime_dense: wire
                    .f_prime_dense
                    .iter()
                    .copied()
                    .map(DenseNodeId::from_raw)
                    .collect(),
                problem: wire.problem.as_ref().map(|p| antecedent_identify::HedgeProblem {
                    variables: vars_from_raw(&p.variables),
                    directed: p.directed.clone().into(),
                    bidirected: p.bidirected.clone().into(),
                    treatments: vars_from_raw(&p.treatments),
                    outcomes: vars_from_raw(&p.outcomes),
                }),
            };
            // A witness that carries its graph is re-checked from the definition on load, so a
            // hand-edited or corrupted hedge cannot pass as a proof of non-identification.
            if certificate.problem.is_some() {
                certificate
                    .verify_carried()
                    .map_err(|e| IoError::Convert(format!("invalid hedge certificate: {e}")))?;
            }
            Some(certificate)
        }
        None => None,
    };
    for estimand in &w.estimands {
        // The estimand's root pointer indexes the arena directly; a dangling
        // one would panic in every consumer that renders or compiles it.
        if estimand.functional as usize >= w.arena.nodes.len() {
            return Err(IoError::Convert(
                "identified estimand functional is outside its expression arena".into(),
            ));
        }
        if let Some(design) = &estimand.rd_design {
            if !design.cutoff.is_finite()
                || !design.bandwidth.is_finite()
                || design.bandwidth <= 0.0
            {
                return Err(IoError::Convert(
                    "sharp RD design needs a finite cutoff and a finite positive bandwidth".into(),
                ));
            }
        }
    }
    Ok(IdentificationResult::from_parts(IdentificationResultParts {
        status,
        query: causal_query_from_wire(&w.query)?,
        estimands: w
            .estimands
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
        arena: expr_arena_from_wire(&w.arena)?,
        derivation: DerivationTrace {
            steps: w
                .derivation
                .iter()
                .map(|s| antecedent_identify::DerivationStep {
                    rule: Arc::from(s.rule.as_str()),
                    detail: Arc::from(s.detail.as_str()),
                })
                .collect(),
        },
        required_assumptions: assumptions_from_wire(&w.required_assumptions)?,
        diagnostics: w
            .diagnostics
            .iter()
            .map(diagnostic_from_wire)
            .collect::<Result<Vec<_>, _>>()?,
        performance: IdentificationPerformanceRecord {
            candidates_examined: w.candidates_examined,
            sets_returned: w.sets_returned,
        },
        hedge,
    }))
}

/// Encode diagnostic.
#[must_use]
pub fn diagnostic_to_wire(d: &Diagnostic) -> DiagnosticWire {
    DiagnosticWire {
        code: d.code.to_string(),
        kind: match d.kind {
            DiagnosticKind::Scientific => "scientific",
            DiagnosticKind::Support => "support",
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
            "support" => DiagnosticKind::Support,
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
mod route_diagnostic_wire_tests {
    use super::{diagnostic_from_wire, diagnostic_to_wire};
    use antecedent_core::diagnostic::{RouteCheck, RouteKind, RouteMass, RouteNativeDiagnostic};

    #[test]
    fn route_records_round_trip_through_the_wire_with_their_mass() {
        let measured = RouteMass::new(0.0625).expect("a finite mass");
        let records = [
            RouteNativeDiagnostic::check(RouteKind::Convergence, RouteCheck::new(false, measured)),
            RouteNativeDiagnostic::check(
                RouteKind::RHat,
                RouteCheck::new(false, RouteMass::zero()),
            ),
            RouteNativeDiagnostic::check(
                RouteKind::PosteriorEss,
                RouteCheck::new(true, RouteMass::unreported()),
            ),
            RouteNativeDiagnostic::unsupported_inference(RouteMass::unreported()),
        ];
        for record in records {
            let diagnostic = record.to_diagnostic();
            let wire = diagnostic_to_wire(&diagnostic);
            let bytes = serde_json::to_vec(&wire).expect("the wire encodes");
            let decoded: super::DiagnosticWire =
                serde_json::from_slice(&bytes).expect("the wire decodes");
            let restored = diagnostic_from_wire(&decoded).expect("the diagnostic decodes");
            assert_eq!(restored, diagnostic);
            assert_eq!(RouteNativeDiagnostic::from_diagnostic(&restored), Some(record));
        }
    }
}
