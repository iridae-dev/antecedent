//! Registered reason codes and the structured details attached to refusals.
//!
//! Refusal codes are a closed registry. `reason_code!` checks a code against it at compile time;
//! `reason_details` carries the machine-readable detail keys each refusal family attaches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Whole `antecedent-core` modules re-exported as namespaces (their items are
/// reached as `reason::<module>::<Item>`).
pub use antecedent_core::{reason_code, reason_details};
