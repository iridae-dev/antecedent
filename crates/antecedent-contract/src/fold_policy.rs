//! Who assigns the folds, whether that assignment is in the receipt, and the
//! dependence the score assumes.
//!
//! One policy. The historical words (`host_owned`, `caller_owned`,
//! `origin_scoped`, `provider_owned`, `none`,
//! `external_learner_owned_not_recorded_by_handoff`, and the learner-route
//! words `iid` and `temporal`) decode onto it. They are not a second parameter.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::named_variants;

/// Who assigns the folds, whether that assignment is in the receipt, and which
/// dependence the score assumes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FoldPolicy {
    /// The host assigns the folds. The assignment is in the receipt. The score
    /// assumes independent rows.
    HostOwned,
    /// The caller assigns the folds. The assignment is in the receipt. The score
    /// assumes independent rows.
    CallerOwned,
    /// Each forecast origin assigns the folds. The assignment is in the receipt.
    /// The score assumes temporal dependence.
    OriginScoped,
    /// The provider assigns the folds. The assignment is in the receipt. The
    /// score assumes independent rows.
    ProviderOwned,
    /// No fold assignment is declared. Nothing is in the receipt. No dependence
    /// is assumed.
    None,
    /// The fold assignment is unknown. The policy is on the receipt the same way
    /// a host-owned policy is; the assignment itself is not. No dependence is
    /// assumed.
    HandoffUnknown,
}

named_variants! {
    FoldPolicy {
        HostOwned => "host_owned",
        CallerOwned => "caller_owned",
        OriginScoped => "origin_scoped",
        ProviderOwned => "provider_owned",
        None => "none",
        HandoffUnknown => "handoff_unknown",
    }
}

impl FoldPolicy {
    /// Who assigns the folds: `host`, `caller`, `origin`, `provider`, `none`,
    /// or `unknown`.
    #[must_use]
    pub const fn assigner(self) -> &'static str {
        match self {
            Self::HostOwned => "host",
            Self::CallerOwned => "caller",
            Self::OriginScoped => "origin",
            Self::ProviderOwned => "provider",
            Self::None => "none",
            Self::HandoffUnknown => "unknown",
        }
    }

    /// Whether the fold assignment itself is in the receipt.
    ///
    /// The handoff's unknown folds are still a value of this enum and are
    /// recorded on the receipt the same way a host-owned policy is. The
    /// assignment is not in the receipt, so this is `false` for that value.
    #[must_use]
    pub const fn assignment_recorded(self) -> bool {
        !matches!(self, Self::None | Self::HandoffUnknown)
    }

    /// Dependence the score assumes: `iid`, `temporal`, or `undeclared`.
    #[must_use]
    pub const fn dependence(self) -> &'static str {
        match self {
            Self::OriginScoped => "temporal",
            Self::None | Self::HandoffUnknown => "undeclared",
            Self::HostOwned | Self::CallerOwned | Self::ProviderOwned => "iid",
        }
    }

    /// Whether a nuisance estimator in this release can honour the policy.
    ///
    /// Only an independent-row score whose fold assignment is in the receipt.
    /// Temporal dependence, an unknown assignment, and a missing assignment are
    /// refused when the learner is declared.
    #[must_use]
    pub const fn estimator_honours(self) -> bool {
        matches!(self, Self::HostOwned | Self::CallerOwned | Self::ProviderOwned)
    }

    /// Decode a legacy fold-policy word onto this enum.
    ///
    /// `external_learner_owned_not_recorded_by_handoff` is [`Self::HandoffUnknown`].
    /// `iid` is [`Self::HostOwned`]. `temporal` is [`Self::OriginScoped`].
    /// The canonical spellings of [`Self::as_str`] decode as themselves.
    #[must_use]
    pub fn from_legacy(word: &str) -> Option<Self> {
        match word {
            "external_learner_owned_not_recorded_by_handoff" => Some(Self::HandoffUnknown),
            "iid" => Some(Self::HostOwned),
            "temporal" => Some(Self::OriginScoped),
            other => Self::from_name(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FoldPolicy;

    #[test]
    fn legacy_words_decode_onto_one_policy_and_round_trip() {
        let handoff =
            FoldPolicy::from_legacy("external_learner_owned_not_recorded_by_handoff").unwrap();
        assert_eq!(handoff, FoldPolicy::HandoffUnknown);
        assert_eq!(handoff.assigner(), "unknown");
        assert!(!handoff.assignment_recorded());
        assert_eq!(handoff.dependence(), "undeclared");
        assert_eq!(handoff.as_str(), "handoff_unknown");
        assert_eq!(FoldPolicy::from_legacy("iid"), Some(FoldPolicy::HostOwned));
        assert_eq!(FoldPolicy::from_legacy("temporal"), Some(FoldPolicy::OriginScoped));
        assert!(!FoldPolicy::OriginScoped.estimator_honours());
        assert!(FoldPolicy::CallerOwned.estimator_honours());
        assert!(FoldPolicy::HostOwned.assignment_recorded());
        for policy in FoldPolicy::ALL {
            assert_eq!(FoldPolicy::from_legacy(policy.as_str()), Some(policy));
        }
        assert_eq!(FoldPolicy::from_legacy("shuffled"), None);
    }
}
