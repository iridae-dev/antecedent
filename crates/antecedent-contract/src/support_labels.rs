//! Support-label consistency of a static response grid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SupportStatus;

use crate::ExternalRefusal;

/// Check a static grid's per-coordinate labels against its value count and summary.
///
/// One label per value coordinate, and `status` must be the worst label. A
/// `MissingEvidence` coordinate is a distinct, weakest label: a summary that hides
/// it behind another label has relabeled missing evidence as a support failure (or
/// as support) and is refused with its own detail.
///
/// # Errors
/// A wrong label count, a summary that is not the worst label, or a relabeled
/// missing-evidence coordinate, each with a distinct registered code and detail.
// A refusal is the cold path of a once-per-artifact check; boxing it would not pay.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_static_point_labels(
    labels: &[SupportStatus],
    status: SupportStatus,
    value_len: usize,
) -> Result<(), ExternalRefusal> {
    let refuse =
        |code: &'static str, detail: &str, expected: String, supplied: String| ExternalRefusal {
            code,
            stage: "support",
            detail: detail.to_owned(),
            offending: None,
            expected: Some(expected),
            supplied: Some(supplied),
            capability: None,
            remedy: Some("label every coordinate and report the worst label as the summary"),
        };
    if labels.len() != value_len {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "coordinate_support.point_count_mismatch",
            value_len.to_string(),
            labels.len().to_string(),
        ));
    }
    let worst = labels.iter().copied().max_by_key(|label| label.severity());
    match worst {
        Some(worst) if worst != status => {
            let hides_missing = worst == SupportStatus::MissingEvidence;
            Err(refuse(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                if hides_missing {
                    "coordinate_support.missing_evidence_relabeled"
                } else {
                    "coordinate_support.summary_not_worst_label"
                },
                worst.as_str().to_owned(),
                status.as_str().to_owned(),
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_labels_must_match_the_count_and_summarize_to_the_worst_label() {
        use SupportStatus as S;
        let labels = [S::Supported, S::WeakOverlap, S::Supported];
        assert_eq!(check_static_point_labels(&labels, S::WeakOverlap, 3), Ok(()));
        let count = check_static_point_labels(&labels, S::WeakOverlap, 4).unwrap_err();
        assert_eq!(
            (count.code, count.detail.as_str()),
            ("invalid_argument", "coordinate_support.point_count_mismatch")
        );
        assert_eq!((count.expected.as_deref(), count.supplied.as_deref()), (Some("4"), Some("3")));
        let summary = check_static_point_labels(&labels, S::Supported, 3).unwrap_err();
        assert_eq!(
            (summary.code, summary.detail.as_str()),
            ("quantity_semantics_mismatch", "coordinate_support.summary_not_worst_label")
        );
        assert_eq!(summary.expected.as_deref(), Some("weak_overlap"));
    }

    #[test]
    fn missing_evidence_is_not_relabeled_as_a_support_failure() {
        use SupportStatus as S;
        let labels = [S::Supported, S::MissingEvidence, S::OutsideEmpiricalSupport];
        // The worst label is missing evidence; reporting zero overlap or an outside
        // coordinate for it hides which kind of failure it is.
        for hidden in [S::OutsideEmpiricalSupport, S::WeakOverlap, S::Supported] {
            let refusal = check_static_point_labels(&labels, hidden, 3).unwrap_err();
            assert_eq!(refusal.detail, "coordinate_support.missing_evidence_relabeled");
            assert_eq!(refusal.expected.as_deref(), Some("missing_evidence"));
            assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code));
        }
        assert_eq!(check_static_point_labels(&labels, S::MissingEvidence, 3), Ok(()));
        // Zero overlap alone stays a support label, distinct from missing evidence.
        let zero = [S::WeakOverlap, S::OutsideEmpiricalSupport];
        assert_eq!(check_static_point_labels(&zero, S::OutsideEmpiricalSupport, 2), Ok(()));
    }
}
