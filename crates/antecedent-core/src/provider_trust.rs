//! One provider-trust vocabulary for every surface that records who stands behind a number.
//!
//! [`ProviderTrust::as_wire`] is the only spelling that is written. Older spellings parse
//! through [`ProviderTrust::from_wire`] and are not emitted.

/// How far a provider's numbers are trusted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProviderTrust {
    /// Implemented and licensed by Antecedent. A caller cannot stamp this on.
    NativeLicensed,
    /// Supplier assertion only. No numerical check was performed here.
    ExternallyAttested,
    /// The exact provider contract passed object-level verification.
    ExactRequestVerified,
    /// No provider verification claim.
    Unverified,
}

impl ProviderTrust {
    /// Canonical wire spelling.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::NativeLicensed => "native_licensed",
            Self::ExternallyAttested => "externally_attested",
            Self::ExactRequestVerified => "verified_extension",
            Self::Unverified => "unverified",
        }
    }

    /// Spelling the distribution artifact still stores.
    ///
    /// `externally_attested` is stored as `external_attested`. Every other level already
    /// matches [`Self::as_wire`].
    #[must_use]
    pub const fn as_distribution_codec(self) -> &'static str {
        match self {
            Self::ExternallyAttested => "external_attested",
            other => other.as_wire(),
        }
    }

    /// Parse a wire spelling, including read-compat aliases.
    ///
    /// Aliases: `external_attested` (attested); `exact_request_verified` and
    /// `external_exact_request_verified` (exact-request verified).
    #[must_use]
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "native_licensed" => Some(Self::NativeLicensed),
            "externally_attested" | "external_attested" => Some(Self::ExternallyAttested),
            "verified_extension" | "exact_request_verified" | "external_exact_request_verified" => {
                Some(Self::ExactRequestVerified)
            }
            "unverified" => Some(Self::Unverified),
            _ => None,
        }
    }
}

impl ProviderTrust {
    /// Trust level of an external scientific object's trust state.
    #[must_use]
    pub fn from_external_state(state: &crate::external::ExternalTrustState) -> Self {
        match state {
            crate::external::ExternalTrustState::NativeLicensed => Self::NativeLicensed,
            crate::external::ExternalTrustState::ExternallyAttested { .. } => {
                Self::ExternallyAttested
            }
            crate::external::ExternalTrustState::ExactRequestVerified(_) => {
                Self::ExactRequestVerified
            }
        }
    }

    /// Trust level retained on a bound external claim.
    #[must_use]
    pub const fn from_bound_level(level: crate::external_binding::BoundTrustLevel) -> Self {
        match level {
            crate::external_binding::BoundTrustLevel::ExternallyAttested => {
                Self::ExternallyAttested
            }
            crate::external_binding::BoundTrustLevel::ExactRequestVerified => {
                Self::ExactRequestVerified
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderTrust;

    #[test]
    fn as_wire_is_unique_and_aliases_parse_back_to_it() {
        let attested = ["externally_attested", "external_attested"];
        let verified =
            ["verified_extension", "exact_request_verified", "external_exact_request_verified"];
        for spelling in attested {
            let trust = ProviderTrust::from_wire(spelling).unwrap();
            assert_eq!(trust, ProviderTrust::ExternallyAttested);
            assert_eq!(trust.as_wire(), "externally_attested");
        }
        for spelling in verified {
            let trust = ProviderTrust::from_wire(spelling).unwrap();
            assert_eq!(trust, ProviderTrust::ExactRequestVerified);
            assert_eq!(trust.as_wire(), "verified_extension");
        }
        assert_eq!(ProviderTrust::ExternallyAttested.as_distribution_codec(), "external_attested");
        assert_eq!(
            ProviderTrust::ExactRequestVerified.as_distribution_codec(),
            "verified_extension"
        );
        assert_eq!(ProviderTrust::from_wire("native"), None);
        assert_eq!(
            ProviderTrust::from_external_state(
                &crate::external::ExternalTrustState::ExternallyAttested { attestor: "lab".into() }
            ),
            ProviderTrust::ExternallyAttested
        );
        assert_eq!(
            ProviderTrust::from_bound_level(
                crate::external_binding::BoundTrustLevel::ExactRequestVerified
            ),
            ProviderTrust::ExactRequestVerified
        );
    }
}
