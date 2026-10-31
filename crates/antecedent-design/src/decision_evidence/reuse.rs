//! Reuse receipt and evidence lineage for composed experimental evidence.
//!
//! Stages form a chain, each digest covering the one before it, so a change invalidates its own
//! stage and everything after it and nothing before it. A decision-rule change reuses the
//! estimate and the derived quantity; a change of the scientific premise reuses nothing.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::CanonicalWriter;

use super::{DecisionFrame, DerivedQuantity, JointExperimentEvidence};

/// A stage of the composed procedure, in dependency order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// What is estimated, in whom, under which contrast, design and exposure assumption.
    Premise,
    /// The pretreatment adjustment applied to each outcome.
    Adjustment,
    /// The fitted joint estimate: observations plus premise plus adjustment.
    Estimate,
    /// The derived quantity: the estimate plus the transformation.
    Derived,
    /// The decision: the derived quantity plus the rule.
    Decision,
}

impl Stage {
    /// All stages in dependency order.
    pub const ALL: [Self; 5] =
        [Self::Premise, Self::Adjustment, Self::Estimate, Self::Derived, Self::Decision];

    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Premise => "premise",
            Self::Adjustment => "adjustment",
            Self::Estimate => "estimate",
            Self::Derived => "derived_quantity",
            Self::Decision => "decision",
        }
    }
}

/// Digest of each stage of one composed procedure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageDigests {
    digests: [String; 5],
}

impl StageDigests {
    /// Digests of the composed procedure.
    #[must_use]
    pub fn of(
        evidence: &JointExperimentEvidence,
        quantity: &DerivedQuantity,
        frame: &DecisionFrame,
    ) -> Self {
        let premise = evidence.scientific_identity();
        let mut adj = CanonicalWriter::new("antecedent.decision_evidence.stage.adjustment.v1");
        adj.text(&premise).text(&evidence.adjustment_identity());
        let adjustment = adj.finish_hex();
        let mut est = CanonicalWriter::new("antecedent.decision_evidence.stage.estimate.v1");
        est.text(&adjustment).text(&evidence.evidence_id);
        let estimate = est.finish_hex();
        let mut der = CanonicalWriter::new("antecedent.decision_evidence.stage.derived.v1");
        der.text(&estimate).text(&quantity.identity());
        let derived = der.finish_hex();
        let mut dec = CanonicalWriter::new("antecedent.decision_evidence.stage.decision.v1");
        dec.text(&derived).text(&frame.rule_identity());
        Self { digests: [premise, adjustment, estimate, derived, dec.finish_hex()] }
    }

    /// Digest of one stage.
    #[must_use]
    pub fn digest(&self, stage: Stage) -> &str {
        &self.digests[stage as usize]
    }
}

/// Whether a stage's previous result can be used again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageStatus {
    /// The digest is unchanged.
    Reused,
    /// The digest changed.
    Invalidated,
}

/// Which stages a change reuses and which it invalidates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReuseReceipt {
    /// Status of each stage in dependency order.
    pub stages: Vec<(Stage, StageStatus)>,
}

impl ReuseReceipt {
    /// Status of one stage.
    #[must_use]
    pub fn status(&self, stage: Stage) -> StageStatus {
        self.stages.iter().find(|(s, _)| *s == stage).map_or(StageStatus::Invalidated, |(_, v)| *v)
    }

    /// Whether anything at all can be reused.
    #[must_use]
    pub fn reuses_any(&self) -> bool {
        self.stages.iter().any(|(_, s)| *s == StageStatus::Reused)
    }
}

/// Compare two composed procedures stage by stage.
#[must_use]
pub fn reuse_receipt(previous: &StageDigests, current: &StageDigests) -> ReuseReceipt {
    ReuseReceipt {
        stages: Stage::ALL
            .iter()
            .map(|s| {
                let status = if previous.digest(*s) == current.digest(*s) {
                    StageStatus::Reused
                } else {
                    StageStatus::Invalidated
                };
                (*s, status)
            })
            .collect(),
    }
}

/// Active or withdrawn status of one piece of evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LineageStatus {
    /// In force.
    Active,
    /// Withdrawn; anything composed from it is no longer supported.
    Withdrawn {
        /// Why.
        reason: String,
    },
}

/// Ledger of evidence and its withdrawal.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvidenceLedger {
    entries: BTreeMap<String, LineageStatus>,
}

impl EvidenceLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record evidence as active.
    pub fn record(&mut self, evidence_id: &str) {
        self.entries.entry(evidence_id.to_owned()).or_insert(LineageStatus::Active);
    }

    /// Withdraw evidence. Returns whether it was known.
    pub fn withdraw(&mut self, evidence_id: &str, reason: &str) -> bool {
        match self.entries.get_mut(evidence_id) {
            Some(s) => {
                *s = LineageStatus::Withdrawn { reason: reason.to_owned() };
                true
            }
            None => false,
        }
    }

    /// Withdrawn parts of a composed evidence id (`a+b` composed from `a` and `b`), with reasons.
    /// Unknown parts are reported as unrecorded: provenance that cannot be traced is not support.
    #[must_use]
    pub fn unsupported_parts(&self, composed_evidence_id: &str) -> Vec<(String, String)> {
        composed_evidence_id
            .split('+')
            .filter_map(|part| match self.entries.get(part) {
                Some(LineageStatus::Active) => None,
                Some(LineageStatus::Withdrawn { reason }) => {
                    Some((part.to_owned(), reason.clone()))
                }
                None => Some((part.to_owned(), "not recorded in the ledger".to_owned())),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::derive::tests::{evidence, margin, no_interference};
    use super::super::*;
    use super::*;

    fn frame(threshold: f64) -> DecisionFrame {
        DecisionFrame { treated_action: "ship".into(), control_action: "hold".into(), threshold }
    }

    #[test]
    fn rule_change_reuses_everything_before_the_decision() {
        let ev = evidence(1.0, no_interference());
        let q = margin(InputBinding::Measured("cost".into()));
        let a = StageDigests::of(&ev, &q, &frame(0.0));
        let b = StageDigests::of(&ev, &q, &frame(1.0));
        let r = reuse_receipt(&a, &b);
        assert_eq!(r.status(Stage::Derived), StageStatus::Reused);
        assert_eq!(r.status(Stage::Decision), StageStatus::Invalidated);
    }

    #[test]
    fn transformation_change_keeps_the_estimate() {
        let ev = evidence(1.0, no_interference());
        let a = StageDigests::of(&ev, &margin(InputBinding::Measured("cost".into())), &frame(0.0));
        let assumed = InputBinding::Assumed {
            lower: 1.0,
            upper: 2.0,
            assumption_id: "a".into(),
            shared_across_arms: true,
        };
        let r = reuse_receipt(&a, &StageDigests::of(&ev, &margin(assumed), &frame(0.0)));
        assert_eq!(r.status(Stage::Estimate), StageStatus::Reused);
        assert_eq!(r.status(Stage::Derived), StageStatus::Invalidated);
    }

    #[test]
    fn premise_change_reuses_nothing_and_new_observations_keep_the_premise() {
        let ev = evidence(1.0, no_interference());
        let q = margin(InputBinding::Measured("cost".into()));
        let a = StageDigests::of(&ev, &q, &frame(0.0));
        let mut other_premise = ev.clone();
        other_premise.horizon = 9;
        assert!(
            !reuse_receipt(&a, &StageDigests::of(&other_premise, &q, &frame(0.0))).reuses_any()
        );
        let mut fresh = ev.clone();
        fresh.evidence_id = "exp-2".into();
        let r = reuse_receipt(&a, &StageDigests::of(&fresh, &q, &frame(0.0)));
        assert_eq!(r.status(Stage::Premise), StageStatus::Reused);
        assert_eq!(r.status(Stage::Adjustment), StageStatus::Reused);
        assert_eq!(r.status(Stage::Estimate), StageStatus::Invalidated);
    }

    #[test]
    fn withdrawal_unsupports_composed_evidence() {
        let mut ledger = EvidenceLedger::new();
        ledger.record("exp-1");
        ledger.record("exp-2");
        assert!(ledger.unsupported_parts("exp-1+exp-2").is_empty());
        assert!(ledger.withdraw("exp-2", "assignment log corrupted"));
        let bad = ledger.unsupported_parts("exp-1+exp-2");
        assert_eq!(bad, vec![("exp-2".to_owned(), "assignment log corrupted".to_owned())]);
        assert_eq!(ledger.unsupported_parts("exp-9").len(), 1);
    }
}
