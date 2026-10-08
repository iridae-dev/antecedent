//! Selection differences and invariances behind each scenario answer
//! (2.3.0 A1 remainder).
//!
//! [`ScenarioSetReport`](crate::transport_scenarios::ScenarioSetReport) says
//! what each scenario answered and the range those answers span. This module says
//! why: for every scenario, the selection differences and the invariances its
//! identified formula relies on (or, for a scenario that did not identify, the
//! structural obstruction), and for each extreme of the structural envelope the
//! same report of the scenario that produced it, so "the answer ranges over
//! 0.35 to 0.56" reads "0.35 from the scenario selecting on `z` (it relies on
//! `P_s(y | do(x), z)` and `P*(z)`), 0.56 from the scenario with no selection
//! (it relies on `P_s(y | do(x))`)".
//!
//! The report is a separate value built from the existing decision and report:
//! no result type and no scenario artifact byte changes. It is a function of the
//! decision alone (the invariances are read from each checked derivation, never
//! recomputed from the evaluated numbers), so a refreshed set with the same
//! decision reports the same invariances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_identify::sid::scenario_invariance::{
    InvarianceReport, InvarianceReportError, invariance_report,
};
use antecedent_identify::sid::scenarios::ScenarioSetDecision;

use crate::error::EstimationError;
use crate::transport_scenarios::{PreparedScenarioSet, ScenarioSetReport};

/// Which end of an envelope an extreme is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeSide {
    /// The smallest identified mean.
    Lower,
    /// The largest identified mean.
    Upper,
}

/// One scenario's invariance report beside its evaluated status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScenarioInvariance {
    /// Scenario name.
    pub name: Arc<str>,
    /// The status the evaluated result reports (it can be `support_failure` or
    /// `unsupported_provider` for a scenario whose invariance report is
    /// still that of its identified formula).
    pub result_status: &'static str,
    /// The scenario's selection differences and invariances or obstruction.
    pub report: InvarianceReport,
}

/// The edges, selections and invariances that produced one envelope extreme.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvelopeExtremeInvariance {
    /// Outcome coordinate of the envelope.
    pub outcome: VariableId,
    /// Which end of the envelope.
    pub side: EnvelopeSide,
    /// The extreme mean.
    pub value: f64,
    /// Scenario attaining it.
    pub scenario: Arc<str>,
    /// That scenario's report.
    pub report: InvarianceReport,
}

/// Invariance reports of a whole scenario set: one per scenario in canonical
/// order, and one per extreme of the structural envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioSetInvarianceReport {
    /// Every scenario, failed as well as successful, in canonical order.
    pub scenarios: Vec<ScenarioInvariance>,
    /// Both extremes of each outcome's envelope; empty when none identified.
    pub extremes: Vec<EnvelopeExtremeInvariance>,
}

impl ScenarioSetInvarianceReport {
    /// The report of the scenario called `name`.
    #[must_use]
    pub fn scenario(&self, name: &str) -> Option<&ScenarioInvariance> {
        self.scenarios.iter().find(|s| &*s.name == name)
    }

    /// The extreme of `outcome` on `side`.
    #[must_use]
    pub fn extreme(
        &self,
        outcome: VariableId,
        side: EnvelopeSide,
    ) -> Option<&EnvelopeExtremeInvariance> {
        self.extremes.iter().find(|e| e.outcome == outcome && e.side == side)
    }

    /// Canonical text of the set: each scenario's name, evaluated status and
    /// report identity, in canonical scenario order, then each extreme.
    #[must_use]
    pub fn canonical_text(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for s in &self.scenarios {
            let _ = writeln!(
                out,
                "scenario={};result={};report={}",
                s.name,
                s.result_status,
                s.report.identity()
            );
        }
        for e in &self.extremes {
            let _ = writeln!(
                out,
                "extreme={};{:?};scenario={};report={}",
                e.outcome.raw(),
                e.side,
                e.scenario,
                e.report.identity()
            );
        }
        out
    }
}

fn refuse(error: &InvarianceReportError) -> EstimationError {
    EstimationError::refused(error.code, format!("{}: {}", error.detail, error.message))
}

/// Build the invariance reports of a decided scenario set beside its evaluated
/// report.
///
/// # Errors
/// A report that does not describe `decision` (different scenario count or
/// names), or an invariance report refused for a derivation that contradicts its
/// own formula.
pub fn scenario_set_invariance_report(
    decision: &ScenarioSetDecision,
    report: &ScenarioSetReport,
) -> Result<ScenarioSetInvarianceReport, EstimationError> {
    if decision.decisions.len() != report.scenarios.len()
        || decision.decisions.iter().zip(&report.scenarios).any(|(d, r)| d.scenario.name != r.name)
    {
        return Err(EstimationError::data_msg(
            "the scenario report does not describe the decided scenario set",
        ));
    }
    let scenarios = decision
        .decisions
        .iter()
        .zip(&report.scenarios)
        .map(|(d, r)| {
            Ok(ScenarioInvariance {
                name: Arc::clone(&r.name),
                result_status: r.status,
                report: invariance_report(&d.scenario, &d.outcome).map_err(|e| refuse(&e))?,
            })
        })
        .collect::<Result<Vec<_>, EstimationError>>()?;
    let mut extremes = Vec::new();
    if let Some(envelope) = &report.envelope {
        for mean in &envelope.means {
            for (side, value, name) in [
                (EnvelopeSide::Lower, mean.lower, &mean.lower_scenario),
                (EnvelopeSide::Upper, mean.upper, &mean.upper_scenario),
            ] {
                let entry = scenarios.iter().find(|s| s.name == *name).ok_or_else(|| {
                    EstimationError::data_msg(format!(
                        "the envelope names scenario {name}, which the report does not hold"
                    ))
                })?;
                extremes.push(EnvelopeExtremeInvariance {
                    outcome: mean.outcome,
                    side,
                    value,
                    scenario: Arc::clone(name),
                    report: entry.report.clone(),
                });
            }
        }
    }
    Ok(ScenarioSetInvarianceReport { scenarios, extremes })
}

impl PreparedScenarioSet {
    /// The selection differences and invariances behind each scenario answer of
    /// `report` (the result of [`Self::evaluate`]), and behind each envelope
    /// extreme. The prepared set, its decision and `report` are not changed.
    ///
    /// # Errors
    /// As [`scenario_set_invariance_report`].
    pub fn invariance_report(
        &self,
        report: &ScenarioSetReport,
    ) -> Result<ScenarioSetInvarianceReport, EstimationError> {
        scenario_set_invariance_report(self.decision(), report)
    }
}
