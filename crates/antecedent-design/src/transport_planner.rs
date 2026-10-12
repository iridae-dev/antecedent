//! Candidate transport-evidence additions and their portable design records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use antecedent_core::EvidenceCatalogDelta;

use crate::{CandidateDesign, DesignCost};

/// One feasible, caller-declared transport evidence addition and its cost.
#[derive(Clone, Debug)]
pub struct TransportEvidenceCandidate {
    /// Stable identity used to report and break equal-cost ties.
    pub id: Arc<str>,
    /// Immutable hypothetical addition. Every regime must remain `Proposed`.
    pub delta: EvidenceCatalogDelta,
    /// Design action represented by the proposed catalog addition.
    pub design: CandidateDesign,
    /// Recruitment and sampling assumptions for this candidate.
    pub recruitment_sampling: Arc<str>,
    /// Explicit feasibility premises; empty means no additional constraints.
    pub feasibility_constraints: Arc<[Arc<str>]>,
    /// Caller-defined cost units and sample budget.
    pub cost: DesignCost,
}

/// Portable candidate-design description using stable numeric identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateDesignRecord {
    /// Stable design kind.
    pub kind: String,
    /// Variable or environment identifiers, depending on kind.
    pub identities: Vec<u32>,
    /// Additional sample or row count when the design declares one.
    pub additional_samples: Option<u64>,
    /// Semantic random-stream tag.
    pub tag: u64,
}

pub(crate) fn candidate_design_record(design: &CandidateDesign) -> CandidateDesignRecord {
    match design {
        CandidateDesign::Measure(plan) => CandidateDesignRecord {
            kind: "measure".into(),
            identities: plan.variables.iter().map(|variable| variable.raw()).collect(),
            additional_samples: None,
            tag: plan.tag,
        },
        CandidateDesign::Intervene(plan) => CandidateDesignRecord {
            kind: "intervene".into(),
            identities: plan.targets.iter().map(|variable| variable.raw()).collect(),
            additional_samples: None,
            tag: plan.tag,
        },
        CandidateDesign::ObserveEnvironment(plan) => CandidateDesignRecord {
            kind: "observe_environment".into(),
            identities: vec![plan.environment.raw()],
            additional_samples: Some(plan.additional_rows),
            tag: plan.tag,
        },
        CandidateDesign::IncreaseSamplingRate(plan) => CandidateDesignRecord {
            kind: "increase_sampling_rate".into(),
            identities: Vec::new(),
            additional_samples: Some(plan.additional_samples),
            tag: plan.tag,
        },
    }
}
