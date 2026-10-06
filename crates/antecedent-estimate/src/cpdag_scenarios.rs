//! Preparation and evaluation of the DAG completions of a supplied CPDAG (2.3A X2).
//!
//! The identify layer enumerates the completions and decides each one through
//! the 2.2 classical catalog route with evidence bound to that completion. This
//! layer compiles each identified completion once against the supplied exact
//! laws through the existing supplied-scenario preparation
//! ([`crate::transport_scenarios`]) and evaluates it. The report is that
//! route's unweighted structural envelope plus a completion receipt naming every
//! completion's identity, edges, status and evidence identity.
//!
//! No mass is assigned to completions: the envelope ranges over the identified
//! completions only, and the identified, unidentified and unevaluated counts
//! are reported separately, never renormalized over the identified members. A
//! completion a budget or cancellation stop never enumerated is counted as
//! unevaluated with its graph unknown, and a stopped result is not exportable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{SearchReceipt, VariableId};
use antecedent_identify::sid::cpdag_completion::{CompletionRecord, CpdagCompletionDecision};

use crate::error::EstimationError;
use crate::transport_scenarios::{
    PreparedScenarioSet, ScenarioSetReport, StatusMass, prepare_transport_scenarios,
};

/// One completion in the receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletionReceiptEntry {
    /// Canonical completion identity (also its scenario name).
    pub id: Arc<str>,
    /// Sorted `(parent, child)` edges of the completion.
    pub edges: Arc<[(VariableId, VariableId)]>,
    /// One of [`crate::transport_scenarios::SCENARIO_STATUSES`].
    pub status: &'static str,
    /// Why the completion did not produce a point, when it did not.
    pub detail: Option<String>,
    /// Identity of the evidence bound to this completion, if any.
    pub evidence_identity: Option<Arc<str>>,
}

/// Every completion, the structural envelope over the identified ones, and the
/// separate counts.
#[derive(Clone, Debug)]
pub struct CpdagScenarioReport {
    /// Identity of the supplied CPDAG.
    pub cpdag_identity: Arc<str>,
    /// Every completion found, in identity order, whatever its status.
    pub completions: Vec<CompletionReceiptEntry>,
    /// Completions a stop left unenumerated (graphs unknown).
    pub not_enumerated: usize,
    /// The supplied-scenario report: statuses with counts, and the unweighted
    /// structural envelope over identified completions. `None` when no
    /// completion was enumerated.
    pub report: Option<ScenarioSetReport>,
    /// Identified completions.
    pub identified: usize,
    /// Completions decided but not identified (every status except
    /// `identified` and `unevaluated`).
    pub unidentified: usize,
    /// Completions left unevaluated, including those never enumerated.
    pub unevaluated: usize,
    /// The first stop's receipt; present means the result is not exportable.
    pub receipt: Option<SearchReceipt>,
}

impl CpdagScenarioReport {
    /// Completions known to exist: found plus not enumerated.
    #[must_use]
    pub fn total(&self) -> usize {
        self.completions.len() + self.not_enumerated
    }

    /// Whether every completion was enumerated and decided.
    #[must_use]
    pub const fn is_exportable(&self) -> bool {
        self.receipt.is_none()
    }

    /// Count of one status, among the completions found.
    #[must_use]
    pub fn status_count(&self, status: &str) -> usize {
        self.completions.iter().filter(|c| c.status == status).count()
    }

    /// Per-status counts of the supplied-scenario report; empty when none was enumerated.
    #[must_use]
    pub fn masses(&self) -> &[StatusMass] {
        self.report.as_ref().map_or(&[], |r| r.masses.as_slice())
    }
}

/// A decided completion set with each identified completion compiled once.
#[derive(Clone, Debug)]
pub struct PreparedCpdagScenarios {
    cpdag_identity: Arc<str>,
    records: Vec<CompletionRecord>,
    not_enumerated: usize,
    receipt: Option<SearchReceipt>,
    inner: Option<PreparedScenarioSet>,
}

/// Compile each identified completion against supplied exact laws through the
/// existing supplied-scenario preparation. Evaluation never re-identifies.
///
/// # Errors
/// As [`prepare_transport_scenarios`]: `schema_mismatch` for a law or request
/// outside the shared coordinate schema, a request or law set that fails for
/// every completion, or cancellation.
pub fn prepare_cpdag_scenarios(
    decision: CpdagCompletionDecision,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<PreparedCpdagScenarios, EstimationError> {
    let CpdagCompletionDecision { cpdag_identity, completions, not_enumerated, decision, receipt } =
        decision;
    let inner = decision
        .map(|decided| prepare_transport_scenarios(decided, data, request, limits, ctx))
        .transpose()?;
    Ok(PreparedCpdagScenarios {
        cpdag_identity,
        records: completions,
        not_enumerated,
        receipt,
        inner,
    })
}

impl PreparedCpdagScenarios {
    /// Identity of the supplied CPDAG.
    #[must_use]
    pub fn cpdag_identity(&self) -> &str {
        &self.cpdag_identity
    }

    /// The prepared supplied-scenario set; `None` when no completion was enumerated.
    #[must_use]
    pub const fn prepared(&self) -> Option<&PreparedScenarioSet> {
        self.inner.as_ref()
    }

    /// Completions found, with their evidence identities, in identity order.
    #[must_use]
    pub fn completions(&self) -> &[CompletionRecord] {
        &self.records
    }

    /// Evaluate every compiled completion and report all of them.
    ///
    /// # Errors
    /// As [`PreparedScenarioSet::evaluate`].
    pub fn evaluate(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<CpdagScenarioReport, EstimationError> {
        let report = self.inner.as_ref().map(|inner| inner.evaluate(ctx)).transpose()?;
        let results = report.as_ref().map_or(&[][..], |r| r.scenarios.as_slice());
        if results.len() != self.records.len() {
            return Err(EstimationError::data_msg(
                "completion receipt and scenario results disagree in length",
            ));
        }
        let mut completions = Vec::with_capacity(results.len());
        for (record, result) in self.records.iter().zip(results) {
            if record.identity != result.name {
                return Err(EstimationError::data_msg(
                    "completion receipt and scenario results disagree on identity",
                ));
            }
            completions.push(CompletionReceiptEntry {
                id: Arc::clone(&record.identity),
                edges: Arc::clone(&record.edges),
                status: result.status,
                detail: result.detail.clone(),
                evidence_identity: record.evidence_identity.clone(),
            });
        }
        let count = |status: &str| completions.iter().filter(|c| c.status == status).count();
        let identified = count("identified");
        let unevaluated_found = count("unevaluated");
        Ok(CpdagScenarioReport {
            cpdag_identity: Arc::clone(&self.cpdag_identity),
            not_enumerated: self.not_enumerated,
            identified,
            unidentified: completions.len() - identified - unevaluated_found,
            unevaluated: unevaluated_found + self.not_enumerated,
            receipt: self.receipt.clone(),
            completions,
            report,
        })
    }
}
