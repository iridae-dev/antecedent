//! Decision-aware evidence planning over a derived quantity.
//!
//! Evidence is ranked by what it would do to the decision, using the existing Normal–Normal
//! [`FutureStudyProgram`] for additional units and a re-evaluation of the decision for a
//! measurement that narrows a declared assumption. The two kinds are reported side by side and
//! are not collapsed into one ranking: expected value under a planning model and a best-case
//! narrowing are not the same quantity.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::EvidenceObligationKind;

use crate::evsi::{
    AffineStateUtility, FutureStudyProgram, NormalMeanFutureData, NormalMeanKnownVarianceUpdate,
    NormalStateBelief, StudyAction,
};

use super::derive::{DecisionFrame, DerivedQuantity, InputBinding, Verdict, decide};
use super::{EvidenceRefusal, JointExperimentEvidence, Request};

/// A feasible piece of next evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum Candidate {
    /// More independently assigned units in each arm, with the same design.
    MoreUnits {
        /// Stable id.
        id: String,
        /// Additional independently assigned units per arm.
        additional_units_per_arm: u64,
        /// Cost in the decision's utility units; `None` is not a zero cost.
        cost: Option<f64>,
    },
    /// More independent complete subjects, pairs or sequences under the same design.
    /// The caller declares that the existing per-unit variance applies to the future study.
    MoreIndependentUnits {
        /// Stable candidate identity.
        id: String,
        /// Number of additional independent subjects, pairs or sequences (not observation rows).
        additional_units: u64,
        /// Identity of the explicit same-design and constant per-unit variance assumption.
        same_design_assumption_id: String,
        /// Cost in decision utility units, if specified.
        cost: Option<f64>,
    },
    /// A measurement expected to narrow a declared assumed range to the given one.
    ResolveAssumption {
        /// Stable id.
        id: String,
        /// Assumption narrowed.
        assumption_id: String,
        /// Range after measuring; equal ends mean the value is pinned.
        narrowed: (f64, f64),
        /// Cost in the decision's utility units.
        cost: Option<f64>,
    },
}

impl Candidate {
    fn id(&self) -> &str {
        match self {
            Self::MoreUnits { id, .. }
            | Self::MoreIndependentUnits { id, .. }
            | Self::ResolveAssumption { id, .. } => id,
        }
    }
}

/// What a candidate would do to the decision.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanValue {
    /// Expected value of acting after the study, under the declared planning model.
    Evsi {
        /// Gross expected value of sample information.
        gross: f64,
        /// Gross minus a declared cost, when one is declared.
        net: Option<f64>,
    },
    /// The decision's classification if the assumption were narrowed as declared. Best case,
    /// not an expectation.
    Resolution {
        /// Classification after narrowing.
        verdict: Verdict,
        /// Whether the narrowed assumption leaves a robust decision.
        resolves: bool,
    },
    /// The candidate could not be evaluated.
    Refused(String),
}

/// One candidate and its value.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanRow {
    /// Candidate id.
    pub id: String,
    /// The obligation kind the candidate would discharge.
    pub kind: EvidenceObligationKind,
    /// Explicit future-design assumption, when the candidate requires one.
    pub planning_assumption_id: Option<String>,
    /// Its value.
    pub value: PlanValue,
}

/// Declared model under which EVSI is computed.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanningModel {
    /// Statement of the model, carried with every plan.
    pub statement: &'static str,
}

/// The plan.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidencePlan {
    /// Additional-unit candidates, best net (or gross) value first.
    pub evsi_rows: Vec<PlanRow>,
    /// Assumption-resolving candidates, resolving ones first.
    pub resolution_rows: Vec<PlanRow>,
    /// The planning model.
    pub model: PlanningModel,
}

/// Rank candidates for a decision.
///
/// # Errors
///
/// [`EvidenceRefusal`] from the underlying decision, or when the decision has no sampling
/// interval to build a planning belief from.
pub fn plan_evidence(
    evidence: &JointExperimentEvidence,
    request: &Request,
    quantity: &DerivedQuantity,
    frame: &DecisionFrame,
    candidates: &[Candidate],
) -> Result<EvidencePlan, EvidenceRefusal> {
    let base = decide(evidence, request, quantity, frame)?;
    let mut evsi_rows = Vec::new();
    let mut resolution_rows = Vec::new();
    for c in candidates {
        match c {
            Candidate::MoreUnits { additional_units_per_arm, cost, .. } => {
                if matches!(
                    evidence.design.assignment_design.as_str(),
                    "matched_pairs"
                        | "switchback"
                        | "temporal_excursion"
                        | "temporal_regime"
                        | "adaptive_regime"
                ) {
                    evsi_rows.push(PlanRow { id: c.id().to_owned(), kind: EvidenceObligationKind::IncreaseSample,
                        planning_assumption_id: None,
                        value: PlanValue::Refused("the more-units planning model has not been specified for matched pairs or temporal assignment sequences".into()) });
                    continue;
                }
                let value = (if base.derived.assumption_range.is_some() {
                    None
                } else {
                    base.derived.sampling.as_ref()
                })
                    .ok_or_else(|| {
                        EvidenceRefusal::Invalid(
                            "planning more units needs a sampling interval without an unresolved assumption range".into(),
                        )
                    })
                    .map(|s| more_units(evidence, s.estimate, s.standard_error, frame, *additional_units_per_arm, *cost));
                evsi_rows.push(PlanRow {
                    id: c.id().to_owned(),
                    kind: EvidenceObligationKind::IncreaseSample,
                    planning_assumption_id: None,
                    value: value.unwrap_or_else(|e| PlanValue::Refused(e.to_string())),
                });
            }
            Candidate::MoreIndependentUnits {
                additional_units,
                same_design_assumption_id,
                cost,
                ..
            } => {
                let value = if same_design_assumption_id.trim().is_empty() {
                    PlanValue::Refused(
                        "a same-design and constant per-unit variance assumption is required"
                            .into(),
                    )
                } else if !matches!(
                    evidence.design.assignment_design.as_str(),
                    "matched_pairs"
                        | "switchback"
                        | "temporal_excursion"
                        | "temporal_regime"
                        | "adaptive_regime"
                ) || evidence.design_details.is_none()
                    || evidence.design.treated_units != evidence.design.control_units
                    || evidence.design.treated_units < 2
                {
                    PlanValue::Refused("independent-unit planning requires a supported retained paired or temporal design with an explicit independent unit count".into())
                } else if base.derived.assumption_range.is_some() {
                    PlanValue::Refused("independent-unit planning cannot collapse an unresolved assumption range into a probability law".into())
                } else if let Some(s) = base.derived.sampling.as_ref() {
                    more_units(
                        evidence,
                        s.estimate,
                        s.standard_error,
                        frame,
                        *additional_units,
                        *cost,
                    )
                } else {
                    PlanValue::Refused(
                        "independent-unit planning requires a supported joint sampling interval"
                            .into(),
                    )
                };
                evsi_rows.push(PlanRow {
                    id: c.id().to_owned(),
                    kind: EvidenceObligationKind::IncreaseSample,
                    planning_assumption_id: Some(same_design_assumption_id.clone()),
                    value,
                });
            }
            Candidate::ResolveAssumption { assumption_id, narrowed, .. } => {
                let mut q = quantity.clone();
                let mut touched = false;
                for b in &mut q.inputs {
                    if let InputBinding::Assumed { lower, upper, assumption_id: id, .. } = b
                        && id == assumption_id
                    {
                        if !narrowed.0.is_finite()
                            || !narrowed.1.is_finite()
                            || narrowed.0 < *lower
                            || narrowed.1 > *upper
                            || narrowed.0 > narrowed.1
                        {
                            return Err(EvidenceRefusal::Invalid(format!(
                                "candidate `{}` must narrow the existing assumption range",
                                c.id()
                            )));
                        }
                        (*lower, *upper) = *narrowed;
                        touched = true;
                    }
                }
                let value = if !touched {
                    PlanValue::Refused(format!("quantity has no assumed input `{assumption_id}`"))
                } else {
                    match decide(evidence, request, &q, frame) {
                        Ok(d) => PlanValue::Resolution {
                            resolves: matches!(d.verdict, Verdict::Robust { .. }),
                            verdict: d.verdict,
                        },
                        Err(e) => PlanValue::Refused(e.to_string()),
                    }
                };
                resolution_rows.push(PlanRow {
                    id: c.id().to_owned(),
                    kind: EvidenceObligationKind::Measure,
                    planning_assumption_id: None,
                    value,
                });
            }
        }
    }
    let key = |r: &PlanRow| match &r.value {
        PlanValue::Evsi { gross, net } => net.unwrap_or(*gross),
        _ => f64::NEG_INFINITY,
    };
    evsi_rows.sort_by(|a, b| key(b).total_cmp(&key(a)).then_with(|| a.id.cmp(&b.id)));
    let resolves = |r: &PlanRow| matches!(r.value, PlanValue::Resolution { resolves: true, .. });
    resolution_rows.sort_by(|a, b| resolves(b).cmp(&resolves(a)).then_with(|| a.id.cmp(&b.id)));
    Ok(EvidencePlan {
        evsi_rows,
        resolution_rows,
        model: PlanningModel {
            statement: "EVSI uses a Normal belief centered on the current contrast with the delta-method sampling variance as its variance, and per-unit variance implied by the current number of independent units; for independent-unit candidates the caller assumes the same design and per-unit variance for new complete independent units; it is a declared planning model, not a calibrated posterior. Resolution rows are best-case narrowings of a declared assumption and are not expectations.",
        },
    })
}

fn more_units(
    evidence: &JointExperimentEvidence,
    contrast: f64,
    standard_error: f64,
    frame: &DecisionFrame,
    additional: u64,
    cost: Option<f64>,
) -> PlanValue {
    let units = evidence.design.treated_units.min(evidence.design.control_units).max(1) as f64;
    let prior_var = standard_error * standard_error;
    let run = || -> Result<(f64, Option<f64>), String> {
        if prior_var <= 0.0 {
            return Err("zero sampling variance: more units cannot change the decision".into());
        }
        let program = FutureStudyProgram::try_new(
            NormalStateBelief::try_new(contrast, prior_var).map_err(|e| e.to_string())?,
            NormalMeanFutureData::try_new(additional, prior_var * units)
                .map_err(|e| e.to_string())?,
            NormalMeanKnownVarianceUpdate,
            vec![
                StudyAction::try_new(&frame.control_action).map_err(|e| e.to_string())?,
                StudyAction::try_new(&frame.treated_action).map_err(|e| e.to_string())?,
            ],
            AffineStateUtility::try_new(vec![0.0, -frame.threshold], vec![0.0, 1.0])
                .map_err(|e| e.to_string())?,
            cost,
        )
        .map_err(|e| e.to_string())?;
        let v = program.evaluate_future_study().map_err(|e| e.to_string())?;
        Ok((v.gross_evsi, v.net_value))
    };
    match run() {
        Ok((gross, net)) => PlanValue::Evsi { gross, net },
        Err(e) => PlanValue::Refused(e),
    }
}

#[cfg(test)]
mod tests {
    use super::super::derive::tests::{evidence, margin, no_interference, request};
    use super::*;

    fn frame() -> DecisionFrame {
        DecisionFrame {
            treated_action: "ship".into(),
            control_action: "hold".into(),
            threshold: 0.0,
        }
    }

    #[test]
    fn more_units_have_value_when_the_decision_is_statistically_open() {
        let ev = evidence(400.0, no_interference());
        let q = margin(InputBinding::Measured("cost".into()));
        let plan = plan_evidence(
            &ev,
            &request(),
            &q,
            &frame(),
            &[
                Candidate::MoreUnits {
                    id: "small".into(),
                    additional_units_per_arm: 10,
                    cost: Some(0.1),
                },
                Candidate::MoreUnits {
                    id: "large".into(),
                    additional_units_per_arm: 400,
                    cost: Some(0.1),
                },
            ],
        )
        .unwrap();
        let PlanValue::Evsi { gross: g_large, net } = plan.evsi_rows[0].value else { panic!() };
        assert_eq!(plan.evsi_rows[0].id, "large");
        assert!(g_large > 0.0 && net.unwrap() < g_large);
    }

    #[test]
    fn narrowing_an_assumption_can_resolve_an_assumption_dependent_decision() {
        let ev = evidence(0.05, no_interference());
        let cost = InputBinding::Assumed {
            lower: 4.0,
            upper: 11.0,
            assumption_id: "a.cost".into(),
            shared_across_arms: true,
        };
        let q = margin(cost);
        let bar = DecisionFrame { threshold: 2.5, ..frame() };
        let plan = plan_evidence(
            &ev,
            &request(),
            &q,
            &bar,
            &[
                Candidate::ResolveAssumption {
                    id: "audit_cost".into(),
                    assumption_id: "a.cost".into(),
                    narrowed: (5.5, 6.5),
                    cost: None,
                },
                Candidate::ResolveAssumption {
                    id: "weak_audit".into(),
                    assumption_id: "a.cost".into(),
                    narrowed: (4.0, 10.0),
                    cost: None,
                },
                Candidate::ResolveAssumption {
                    id: "wrong".into(),
                    assumption_id: "a.nope".into(),
                    narrowed: (1.0, 1.0),
                    cost: None,
                },
            ],
        )
        .unwrap();
        assert_eq!(plan.resolution_rows[0].id, "audit_cost");
        assert!(matches!(
            plan.resolution_rows[0].value,
            PlanValue::Resolution { resolves: true, .. }
        ));
        assert!(matches!(
            plan.resolution_rows[1].value,
            PlanValue::Resolution { resolves: false, .. }
        ));
        assert!(matches!(plan.resolution_rows[2].value, PlanValue::Refused(_)));
    }
}
