//! The facade's conversion from the shared `not_supported` refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CausalError;

use super::cell::SupportRefusal;

impl From<antecedent_core::NotSupported> for CausalError {
    fn from(refusal: antecedent_core::NotSupported) -> Self {
        Self::Support { id: SupportRefusal::NotSupported, message: refusal.message() }
    }
}
