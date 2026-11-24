//! `antecedent_contract::check_static_point_labels` accepts a static grid's
//! summary only when it is the worst coordinate label, so a summary never reads
//! better than a coordinate it summarizes.
//!
//! Every grid of up to `MAX_LABELS` labels is covered. The Lean statement of the
//! same rule is `Antecedent.Revalidation.labelsAccepted_summary`.

use antecedent_contract::check_static_point_labels;
use antecedent_core::SupportStatus;

const MAX_LABELS: usize = 4;

const STATUSES: [SupportStatus; 5] = [
    SupportStatus::Supported,
    SupportStatus::WeakOverlap,
    SupportStatus::Extrapolative,
    SupportStatus::OutsideEmpiricalSupport,
    SupportStatus::MissingEvidence,
];

fn any_status() -> SupportStatus {
    let index: usize = kani::any();
    kani::assume(index < STATUSES.len());
    STATUSES[index]
}

fn any_labels() -> ([SupportStatus; MAX_LABELS], usize) {
    let labels = [any_status(), any_status(), any_status(), any_status()];
    let len: usize = kani::any();
    kani::assume(len <= MAX_LABELS);
    (labels, len)
}

/// The most severe label, computed independently of the checked function.
fn worst(labels: &[SupportStatus]) -> Option<SupportStatus> {
    let mut out: Option<SupportStatus> = None;
    for label in labels {
        out = match out {
            Some(w) if w.severity() >= label.severity() => Some(w),
            _ => Some(*label),
        };
    }
    out
}

/// An accepted summary is at least as severe as every label and is one of them.
#[kani::proof]
#[kani::unwind(6)]
fn accepted_summary_is_never_better_than_a_label() {
    let (labels, len) = any_labels();
    let labels = &labels[..len];
    let status = any_status();
    if check_static_point_labels(labels, status, len).is_ok() {
        for label in labels {
            assert!(label.severity() <= status.severity());
        }
        assert!(labels.is_empty() || labels.contains(&status));
    }
}

/// The worst label with the right count is accepted: the check refuses nothing it should accept.
#[kani::proof]
#[kani::unwind(6)]
fn worst_label_summary_is_accepted() {
    let (labels, len) = any_labels();
    let labels = &labels[..len];
    if let Some(status) = worst(labels) {
        assert!(check_static_point_labels(labels, status, len).is_ok());
    }
}

/// A grid with a missing-evidence coordinate is summarized as missing evidence or refused.
#[kani::proof]
#[kani::unwind(6)]
fn missing_evidence_is_never_hidden() {
    let (labels, len) = any_labels();
    let labels = &labels[..len];
    let status = any_status();
    if labels.contains(&SupportStatus::MissingEvidence) && status != SupportStatus::MissingEvidence
    {
        assert!(check_static_point_labels(labels, status, len).is_err());
    }
}
