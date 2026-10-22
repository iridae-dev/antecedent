//! Mixed-module policy for a release workflow.
//!
//! Shared probabilistic modules are one coherent joint model, or a modular
//! alternative that records how shared parameters and priors were reconciled
//! and states a [`CutFeedback`] policy. [`CutFeedback::Cut`] and
//! [`CutFeedback::Feedback`] are both named; the modular constructor takes one
//! of them as an argument.
//!
//! The types here are the policy record. Fitting a model, melding posteriors,
//! and producing a posterior are outside this module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{error::Error, fmt, sync::Arc};

use antecedent_core::named_variants;
use antecedent_core::text::is_blank;

/// How a release workflow treats shared probabilistic modules.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
#[must_use = "a mixed-module policy records the release workflow's modeling choice"]
pub enum MixedModulePolicy {
    /// One coherent joint model.
    ///
    /// Shared parameters and priors belong to that model. No cut or feedback
    /// policy is stored.
    CoherentJoint,
    /// A modular alternative with a reconciliation record and a named cut or
    /// feedback policy.
    DeclaredModular {
        /// Non-blank shared-parameter and prior reconciliation.
        reconciliation: ReconciliationRecord,
        /// The cut or feedback policy the caller named.
        cut_feedback: CutFeedback,
    },
}

impl MixedModulePolicy {
    /// Coherent joint modeling, with no cut stored.
    pub const fn coherent_joint() -> Self {
        Self::CoherentJoint
    }

    /// A modular policy that keeps `reconciliation` and `cut_feedback`.
    ///
    /// # Errors
    ///
    /// [`MixedModuleRefusal::BlankReconciliation`] when `reconciliation` is
    /// empty or only whitespace. That result contains the refusal alone.
    pub fn declared_modular(
        reconciliation: &str,
        cut_feedback: CutFeedback,
    ) -> Result<Self, MixedModuleRefusal> {
        Ok(Self::DeclaredModular {
            reconciliation: ReconciliationRecord::try_new(reconciliation)?,
            cut_feedback,
        })
    }

    /// Cut or feedback stored on a modular policy.
    ///
    /// Coherent joint modeling returns [`None`].
    #[must_use]
    pub const fn cut_feedback(&self) -> Option<CutFeedback> {
        match self {
            Self::CoherentJoint => None,
            Self::DeclaredModular { cut_feedback, .. } => Some(*cut_feedback),
        }
    }

    /// Shared-parameter and prior reconciliation stored on a modular policy.
    ///
    /// Coherent joint modeling returns [`None`].
    #[must_use]
    pub const fn reconciliation(&self) -> Option<&ReconciliationRecord> {
        match self {
            Self::CoherentJoint => None,
            Self::DeclaredModular { reconciliation, .. } => Some(reconciliation),
        }
    }
}

/// Cut or feedback for a declared modular alternative.
///
/// Both alternatives are explicit arguments. [`MixedModulePolicy::coherent_joint`]
/// stores neither.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CutFeedback {
    /// The modules are cut: the caller named the absence of feedback.
    Cut,
    /// The modules feed back; the reconciliation record states how.
    Feedback,
}

named_variants! {
    CutFeedback {
        Cut => "cut",
        Feedback => "feedback",
    }
}

impl fmt::Display for CutFeedback {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Caller's account of how shared parameters and their priors were reconciled.
///
/// The text is the record. It is produced only from non-blank text.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ReconciliationRecord {
    text: Arc<str>,
}

impl ReconciliationRecord {
    /// Keep `text` as the reconciliation record.
    ///
    /// # Errors
    ///
    /// [`MixedModuleRefusal::BlankReconciliation`] when `text` is empty or
    /// only whitespace.
    pub fn try_new(text: &str) -> Result<Self, MixedModuleRefusal> {
        if is_blank(text) || is_invisible(text) {
            Err(MixedModuleRefusal::BlankReconciliation)
        } else {
            Ok(Self { text: Arc::from(text) })
        }
    }

    /// The reconciliation text the caller supplied.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// Text made only of whitespace, control, or zero-width format characters.
///
/// `str::trim` treats Unicode white space as blank but not a zero-width space or
/// joiner, which render as nothing. A record of those characters says nothing.
fn is_invisible(text: &str) -> bool {
    text.chars().all(|c| {
        c.is_whitespace()
            || c.is_control()
            || matches!(c, '\u{200b}'..='\u{200f}' | '\u{2060}' | '\u{feff}')
    })
}

impl fmt::Display for ReconciliationRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal to build a modular policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
#[must_use = "a refusal is the failed modular construction"]
pub enum MixedModuleRefusal {
    /// The modular declaration had a blank reconciliation record.
    BlankReconciliation,
}

impl MixedModuleRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::BlankReconciliation => antecedent_core::reason_code!("invalid_argument"),
        }
    }

    /// Stable `mixed_module.*` detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::BlankReconciliation => "mixed_module.blank_reconciliation",
        }
    }
}

impl fmt::Display for MixedModuleRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::BlankReconciliation => {
                "a declared modular policy needs a non-blank shared-parameter and prior reconciliation record"
            }
        };
        write!(
            formatter,
            "{}{}: {}: {message}",
            antecedent_core::reason_code::PREFIX,
            self.code(),
            self.detail(),
        )
    }
}

impl Error for MixedModuleRefusal {}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{CutFeedback, MixedModulePolicy, MixedModuleRefusal, ReconciliationRecord};

    #[test]
    fn coherent_joint_stores_no_cut_or_reconciliation() {
        let policy = MixedModulePolicy::coherent_joint();
        assert_eq!(policy.cut_feedback(), None);
        assert!(policy.reconciliation().is_none());
    }

    #[test]
    fn text_that_renders_as_nothing_is_not_a_reconciliation() {
        let blanks = [
            "",
            " ",
            "\t\n",
            "\u{a0}",
            "\u{2003}\u{3000}",
            "\u{200b}",
            "\u{200d}\u{feff}",
            " \u{2060} ",
            "\u{0}",
        ];
        for blank in blanks {
            for policy in CutFeedback::ALL {
                assert_eq!(
                    MixedModulePolicy::declared_modular(blank, policy),
                    Err(MixedModuleRefusal::BlankReconciliation),
                    "{blank:?}"
                );
            }
            assert_eq!(
                ReconciliationRecord::try_new(blank),
                Err(MixedModuleRefusal::BlankReconciliation)
            );
        }
        // One visible character is enough, wherever it sits among invisible ones.
        for record in ["x", "\u{200b}x", " x ", "\u{a0}\u{200d}é\u{feff}"] {
            assert!(ReconciliationRecord::try_new(record).is_ok(), "{record:?}");
        }
    }

    #[test]
    fn the_record_keeps_the_callers_text_without_trimming() {
        let record = "  shared theta;\n prior product renormalized on the overlap \t";
        let policy = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();
        let kept = policy.reconciliation().unwrap();
        assert_eq!(kept.as_str(), record);
        assert_eq!(kept.to_string(), record);
        // Reconciliations that differ only in padding are different records.
        let trimmed = MixedModulePolicy::declared_modular(record.trim(), CutFeedback::Cut).unwrap();
        assert_ne!(policy, trimmed);
    }

    #[test]
    fn policies_differ_by_cut_feedback_and_by_record_and_hash_accordingly() {
        let record = "shared theta; prior product renormalized on the overlap";
        let cut = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();
        let feedback = MixedModulePolicy::declared_modular(record, CutFeedback::Feedback).unwrap();
        let other_record =
            MixedModulePolicy::declared_modular("pooled theta", CutFeedback::Cut).unwrap();
        let same_as_cut = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();

        assert_eq!(cut, same_as_cut);
        let distinct = [&cut, &feedback, &other_record, &MixedModulePolicy::CoherentJoint];
        let set: HashSet<&MixedModulePolicy> = distinct.into_iter().collect();
        assert_eq!(set.len(), 4);
        assert!(set.contains(&same_as_cut));
        assert_eq!(cut.cut_feedback(), Some(CutFeedback::Cut));
        assert_eq!(feedback.cut_feedback(), Some(CutFeedback::Feedback));
    }

    #[test]
    fn cut_feedback_names_parse_exactly() {
        for policy in CutFeedback::ALL {
            assert_eq!(CutFeedback::from_name(policy.as_str()), Some(policy));
            assert_eq!(policy.to_string(), policy.as_str());
            // Neither case variants nor padding are the same spelling.
            assert_eq!(CutFeedback::from_name(&policy.as_str().to_uppercase()), None);
            assert_eq!(CutFeedback::from_name(&format!(" {} ", policy.as_str())), None);
        }
        assert_ne!(CutFeedback::Cut.as_str(), CutFeedback::Feedback.as_str());
        assert_eq!(CutFeedback::from_name(""), None);
        assert_eq!(CutFeedback::from_name("cut_feedback"), None);
    }

    #[test]
    fn a_refusal_message_carries_the_registered_code_and_detail_it_reports() {
        let refusal = MixedModulePolicy::declared_modular(" ", CutFeedback::Feedback).unwrap_err();
        let message = refusal.to_string();
        assert!(message.starts_with(antecedent_core::reason_code::PREFIX), "{message}");
        assert!(message.contains(refusal.code()), "{message}");
        assert!(message.contains(refusal.detail()), "{message}");
        assert!(message.contains("reconciliation"), "{message}");
        assert_eq!(refusal.code(), antecedent_core::reason_code!("invalid_argument"));
        assert!(refusal.detail().starts_with("mixed_module."));
        // The refusal is an error value usable through the standard trait object.
        let boxed: Box<dyn std::error::Error> = Box::new(refusal);
        assert_eq!(boxed.to_string(), message);
    }
}
