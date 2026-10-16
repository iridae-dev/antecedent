//! Typed refusals of selective discovery: a registered reason code, a namespaced detail and
//! the declaration, evidence or action that would license the request.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::DiscoveryError;

/// Why a selective-discovery request was refused or failed.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SelectiveDiscoveryError {
    /// The underlying discovery algorithm failed.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// A scientific or declaration refusal.
    #[error("{detail}: {remedy}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// Namespaced `selective_discovery.*` detail.
        detail: &'static str,
        /// What would license the request.
        remedy: String,
    },
}

impl SelectiveDiscoveryError {
    /// A refusal with a registered code.
    #[must_use]
    pub fn refused(code: &'static str, detail: &'static str, remedy: impl Into<String>) -> Self {
        Self::Refused { code, detail, remedy: remedy.into() }
    }

    /// The registered reason code, when this is a refusal.
    #[must_use]
    pub const fn code(&self) -> Option<&'static str> {
        match self {
            Self::Refused { code, .. } => Some(*code),
            Self::Discovery(_) => None,
        }
    }

    /// The namespaced detail, when this is a refusal.
    #[must_use]
    pub const fn detail(&self) -> Option<&'static str> {
        match self {
            Self::Refused { detail, .. } => Some(*detail),
            Self::Discovery(_) => None,
        }
    }
}
