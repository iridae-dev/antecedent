//! Seed indexes inside [`antecedent_core::StreamDomain::Learner`].
//!
//! The numbers are part of fitted seeds. Renaming a call site without keeping the index
//! would refit every model that draws from that stream.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Gradient-boosted tree initialization.
pub(crate) const GBT: u64 = 1;

/// Random-forest / extra-trees initialization.
pub(crate) const FOREST: u64 = 2;

/// Cross-fit fold shuffle.
pub(crate) const FOLD_SHUFFLE: u64 = 0xF01D;

/// Burn network initialization, mixed once more by the learner before `MlpSpec.seed`.
pub(crate) const NEURAL: u64 = 0x4E45_5500;
