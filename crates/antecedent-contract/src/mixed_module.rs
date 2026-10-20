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

impl CutFeedback {
    /// Both policies, in declaration order.
    pub const ALL: [Self; 2] = [Self::Cut, Self::Feedback];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cut => "cut",
            Self::Feedback => "feedback",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    ///
    /// Any other string, including a blank one, is [`None`].
    #[must_use = "an unknown name is not a cut or feedback policy"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|policy| policy.as_str() == name)
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
        if is_blank(text) {
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
    use super::{CutFeedback, MixedModulePolicy, MixedModuleRefusal};

    #[test]
    fn coherent_joint_needs_no_cut() {
        let policy = MixedModulePolicy::coherent_joint();
        assert_eq!(policy, MixedModulePolicy::CoherentJoint);
        assert_eq!(policy.cut_feedback(), None);
        assert!(policy.reconciliation().is_none());
    }

    #[test]
    fn modular_without_reconciliation_is_refused() {
        for blank in ["", " ", "\t\n"] {
            let refused = MixedModulePolicy::declared_modular(blank, CutFeedback::Cut);
            assert_eq!(refused, Err(MixedModuleRefusal::BlankReconciliation));
            let refusal = refused.unwrap_err();
            assert_eq!(refusal.code(), antecedent_core::reason_code!("invalid_argument"));
            assert_eq!(refusal.detail(), "mixed_module.blank_reconciliation");
        }
    }

    #[test]
    fn modular_with_reconciliation_retains_the_cut_feedback_choice() {
        let record = "shared theta; prior product renormalized on the overlap";
        let cut = MixedModulePolicy::declared_modular(record, CutFeedback::Cut).unwrap();
        let feedback = MixedModulePolicy::declared_modular(record, CutFeedback::Feedback).unwrap();

        assert_eq!(cut.cut_feedback(), Some(CutFeedback::Cut));
        assert_eq!(feedback.cut_feedback(), Some(CutFeedback::Feedback));
        assert_eq!(cut.reconciliation().map(super::ReconciliationRecord::as_str), Some(record));
        assert_eq!(
            feedback.reconciliation().map(super::ReconciliationRecord::as_str),
            Some(record)
        );
        assert_ne!(cut, feedback);
        assert_ne!(cut, MixedModulePolicy::coherent_joint());
        assert_eq!(CutFeedback::from_name("cut"), Some(CutFeedback::Cut));
        assert_eq!(CutFeedback::from_name("feedback"), Some(CutFeedback::Feedback));
        assert_eq!(CutFeedback::from_name(""), None);
    }
}
