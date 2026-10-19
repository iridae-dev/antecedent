//! Provider replay declarations.
//!
//! [`ProviderReplay::fresh_against`] returns [`ReplayFreshness::Fresh`] only when
//! the semantic seed, state digest, and provider version are all present and equal
//! and the function identity and column-list digest match. If the seed, state, or
//! version is absent on either declaration, the result is
//! [`ReplayFreshness::OpaqueInvalidation`], including when the function identity
//! and the column list are unchanged. A missing seed is not seed zero. A changed
//! seed with the same function and columns is [`ReplayFreshness::Stale`].
//!
//! The comparison reads two declarations. It does not cache a result and it does
//! not call a provider.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SemanticDigest;

/// Whether a provider declaration still answers a previous declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub enum ReplayFreshness {
    /// Seed, state, and provider version are present and equal, and the function
    /// identity and column-list digest match.
    Fresh,
    /// Both declarations are complete, and the seed, state, version, function,
    /// or column list differs.
    Stale,
    /// Seed, state, or provider version is absent, so the result cannot be shown
    /// current.
    OpaqueInvalidation,
}

/// Declared dependencies of one provider replay.
///
/// `seed`, `state`, and `provider_version` may be absent. Absence is not a zero
/// seed, an empty state, or version zero. `function` is the function-object
/// identity. `column_list` is the digest of the columns the provider read.
///
/// Holding this value stores no callback result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ProviderReplay {
    /// Identity of the function object. An unchanged function does not keep a
    /// result fresh.
    pub function: SemanticDigest,
    /// Digest of the column list. An unchanged column list does not keep a
    /// result fresh.
    pub column_list: SemanticDigest,
    /// Semantic seed of the provider stream. `None` is not seed zero.
    pub seed: Option<u64>,
    /// Digest of provider state. `None` is not an empty state.
    pub state: Option<SemanticDigest>,
    /// Provider version. `None` is not version zero.
    pub provider_version: Option<u64>,
}

impl ProviderReplay {
    /// Compare `self` with `previous`.
    ///
    /// Absent seed, state, or version on either side is
    /// [`ReplayFreshness::OpaqueInvalidation`]. Present values are fresh only
    /// when seed, state, version, function identity, and column list are equal.
    /// Any other complete difference is [`ReplayFreshness::Stale`].
    pub const fn fresh_against(&self, previous: &Self) -> ReplayFreshness {
        let Some(current) = declared(self) else {
            return ReplayFreshness::OpaqueInvalidation;
        };
        let Some(prior) = declared(previous) else {
            return ReplayFreshness::OpaqueInvalidation;
        };
        if same_declared(current, prior)
            && same_digest(self.function, previous.function)
            && same_digest(self.column_list, previous.column_list)
        {
            ReplayFreshness::Fresh
        } else {
            ReplayFreshness::Stale
        }
    }
}

/// Byte equality, so the comparison can stay a const fn on Rust 1.85.
const fn same_digest(left: SemanticDigest, right: SemanticDigest) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut index = 0;
    while index < 32 {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

const fn same_declared(
    left: (u64, SemanticDigest, u64),
    right: (u64, SemanticDigest, u64),
) -> bool {
    left.0 == right.0 && same_digest(left.1, right.1) && left.2 == right.2
}

/// Seed, state, and version when every one of them was declared.
const fn declared(replay: &ProviderReplay) -> Option<(u64, SemanticDigest, u64)> {
    match (replay.seed, replay.state, replay.provider_version) {
        (Some(seed), Some(state), Some(version)) => Some((seed, state, version)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    const fn complete(seed: u64) -> ProviderReplay {
        ProviderReplay {
            function: digest(1),
            column_list: digest(2),
            seed: Some(seed),
            state: Some(digest(3)),
            provider_version: Some(1),
        }
    }

    #[test]
    fn complete_equal_declaration_is_fresh() {
        let previous = complete(4);
        assert_eq!(previous.fresh_against(&previous), ReplayFreshness::Fresh);
        assert_eq!(complete(0).fresh_against(&complete(0)), ReplayFreshness::Fresh);
    }

    #[test]
    fn same_function_and_columns_with_a_new_seed_is_stale() {
        let previous = complete(1);
        let current = complete(2);
        assert_eq!(current.function, previous.function);
        assert_eq!(current.column_list, previous.column_list);
        assert_eq!(current.state, previous.state);
        assert_eq!(current.provider_version, previous.provider_version);
        assert_eq!(current.fresh_against(&previous), ReplayFreshness::Stale);
    }

    #[test]
    fn missing_version_is_opaque_invalidation() {
        let previous = complete(9);
        let missing = ProviderReplay { provider_version: None, ..previous };
        assert_eq!(missing.function, previous.function);
        assert_eq!(missing.column_list, previous.column_list);
        assert_eq!(missing.seed, previous.seed);
        assert_eq!(missing.state, previous.state);
        assert_eq!(missing.fresh_against(&previous), ReplayFreshness::OpaqueInvalidation);
        assert_eq!(previous.fresh_against(&missing), ReplayFreshness::OpaqueInvalidation);
        assert_eq!(missing.fresh_against(&missing), ReplayFreshness::OpaqueInvalidation);
    }

    #[test]
    fn missing_seed_is_not_treated_as_seed_zero() {
        let zero = complete(0);
        let missing = ProviderReplay { seed: None, ..zero };
        assert_eq!(zero.fresh_against(&zero), ReplayFreshness::Fresh);
        assert_eq!(missing.fresh_against(&zero), ReplayFreshness::OpaqueInvalidation);
        assert_eq!(zero.fresh_against(&missing), ReplayFreshness::OpaqueInvalidation);
        assert_eq!(missing.fresh_against(&missing), ReplayFreshness::OpaqueInvalidation);
        let other = ProviderReplay { seed: None, ..complete(7) };
        assert_eq!(missing.fresh_against(&other), ReplayFreshness::OpaqueInvalidation);
    }

    #[test]
    fn missing_state_is_opaque_invalidation() {
        let previous = complete(3);
        let missing = ProviderReplay { state: None, ..previous };
        assert_eq!(missing.function, previous.function);
        assert_eq!(missing.column_list, previous.column_list);
        assert_eq!(missing.fresh_against(&previous), ReplayFreshness::OpaqueInvalidation);
        assert_eq!(missing.fresh_against(&missing), ReplayFreshness::OpaqueInvalidation);
    }

    #[test]
    fn changed_function_or_columns_is_stale() {
        let previous = complete(5);
        let other_function = ProviderReplay { function: digest(9), ..previous };
        let other_columns = ProviderReplay { column_list: digest(8), ..previous };
        assert_eq!(other_function.fresh_against(&previous), ReplayFreshness::Stale);
        assert_eq!(other_columns.fresh_against(&previous), ReplayFreshness::Stale);
    }
}
