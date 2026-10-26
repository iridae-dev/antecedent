//! Identification product retained when a result is lowered.
//!
//! [`IdentificationResult`] is the strategy output. This record is what the
//! compiler keeps after that result is lowered: the functional or identified
//! set (estimands and the arena that owns their expressions), the derivation,
//! the assumptions, the required factors, the unresolved obligations, and the
//! bounded-search diagnostics. Search truncation stays an execution fact.
//! A finite number stored beside a result is numerical feasibility, not
//! identification.
//!
//! [`PartialEq`] and [`CompiledIdentificationProduct::identity`] cover those
//! scientific fields. They omit bounded-search diagnostics and stored
//! numerical values, so two runs that differ only by budget or by a feasible
//! number are the same identification claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use thiserror::Error;

use antecedent_core::{
    AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource, AssumptionStatus,
    CanonicalEncode, CanonicalWriter, Diagnostic, IdentificationStatus, ScientificTarget,
    ScientificWorldId, SemanticDigest, TargetForm, VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprListId, ExprNode, IdentifiedEstimand,
    InterventionSetId, PopulationKeyId, RdDesignParams, VarSetId,
};

use crate::result::IdentificationResult;

mod product;
mod render;
#[cfg(test)]
mod tests;
mod types;
mod validate;

pub use product::*;
pub use types::*;
#[allow(unused_imports, reason = "sibling submodules reach these helpers through the parent")]
use {render::*, validate::*};
