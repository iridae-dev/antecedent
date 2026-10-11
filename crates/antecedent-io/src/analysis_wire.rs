//! Analysis / estimate / identification / refutation / diagnostic wire types.
//!
//! The identification and diagnostic half lives in `antecedent-io-core`; the
//! effect-estimate and refutation half needs the estimators. Both are public
//! here under the one path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::analysis_wire::*;

pub use crate::estimate_wire::*;
