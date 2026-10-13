//! Structural envelope over declared unseen mass.
//!
//! When tables have unseen cells and the caller declares a mass `delta` of each
//! such table as sitting on its unseen cells, the table is no longer determined.
//! The members evaluated here are the finite family in which each table either
//! keeps its observed shares or moves `delta` of its mass onto exactly one of
//! its unseen cells (scaling the observed shares by `1 - delta`). Each member is
//! evaluated through the stored derivation; the envelope is the least and the
//! greatest defined member. It is a range over that family, not an interval,
//! not a point, and not a proven bound over every distribution of the declared
//! mass over the unseen cells: the functional is a ratio of sums of products, so
//! the extremes over the full polytope need not sit at a member.
//!
//! A member whose evaluation is undefined (a conditioning event still without
//! mass) is counted and reported; if no member is defined the request is
//! refused. Members are never dropped silently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_identify::CounterfactualTransportDerivation;

use super::counts::{LawFlavor, SampledRegimeCounts};
use super::{
    DETAIL_ENVELOPE_BUDGET, DETAIL_ENVELOPE_EMPTY, SampledTransportLimits, evaluate_probability,
    refused,
};
use crate::error::EstimationError;

/// The envelope of a declared unseen-mass family.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralEnvelope {
    /// Least defined member value.
    pub lower: f64,
    /// Greatest defined member value.
    pub upper: f64,
    /// Declared unseen mass per table with unseen cells.
    pub unseen_mass: f64,
    /// Members of the family.
    pub members: u64,
    /// Members whose evaluation was defined.
    pub defined_members: u64,
    /// Members that were undefined (zero support), counted and not dropped.
    pub undefined_members: u64,
    /// Tables that carry unseen cells.
    pub tables_with_unseen_cells: usize,
    /// Always `structural_envelope_vertex_family`.
    pub label: &'static str,
}

/// Label of the envelope: a range over a finite family of fillings.
pub const STRUCTURAL_ENVELOPE_LABEL: &str = "structural_envelope_vertex_family";

/// Evaluate the family. `probabilities` are the observed shares of each table.
///
/// # Errors
///
/// `transport_budget_cancel` for a family larger than the declared limit,
/// `transport_support_failure` when no member is defined, and any evaluation
/// failure other than an undefined member.
pub(crate) fn evaluate_envelope(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
    probabilities: &[Vec<f64>],
    unseen_mass: f64,
    limits: &SampledTransportLimits,
    ctx: &ExecutionContext,
) -> Result<StructuralEnvelope, EstimationError> {
    let unseen: Vec<Vec<usize>> = tables.iter().map(SampledRegimeCounts::unseen_cells).collect();
    let mut members: u64 = 1;
    for cells in &unseen {
        members = members.saturating_mul(1 + cells.len() as u64);
    }
    if members > u64::from(limits.max_envelope_evaluations) {
        return Err(refused(
            antecedent_core::reason_code!("transport_budget_cancel"),
            DETAIL_ENVELOPE_BUDGET,
            format!(
                "the unseen-mass family has {members} members; the declared limit is {}",
                limits.max_envelope_evaluations
            ),
        ));
    }
    let flavors = vec![LawFlavor::Model; tables.len()];
    let mut choice = vec![0usize; tables.len()];
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut defined = 0u64;
    let mut undefined = 0u64;
    loop {
        let trial: Vec<Vec<f64>> = (0..tables.len())
            .map(|t| {
                let mut p = probabilities[t].clone();
                if choice[t] > 0 {
                    let cell = unseen[t][choice[t] - 1];
                    for q in &mut p {
                        *q *= 1.0 - unseen_mass;
                    }
                    p[cell] += unseen_mass;
                }
                p
            })
            .collect();
        match evaluate_probability(derivation, tables, &trial, &flavors, limits, ctx) {
            Ok(value) => {
                defined += 1;
                lower = lower.min(value);
                upper = upper.max(value);
            }
            Err(EstimationError::Refused { code, .. })
                if code == antecedent_core::reason_code!("transport_support_failure")
                    || code == antecedent_core::reason_code!("transport_missing_provider") =>
            {
                undefined += 1;
            }
            Err(other) => return Err(other),
        }
        let mut t = tables.len();
        loop {
            if t == 0 {
                if defined == 0 {
                    return Err(refused(
                        antecedent_core::reason_code!("transport_support_failure"),
                        DETAIL_ENVELOPE_EMPTY,
                        "no member of the unseen-mass family has a defined value: the declared \
                         unseen mass does not reach the zero-mass events the formula conditions on",
                    ));
                }
                return Ok(StructuralEnvelope {
                    lower,
                    upper,
                    unseen_mass,
                    members,
                    defined_members: defined,
                    undefined_members: undefined,
                    tables_with_unseen_cells: unseen.iter().filter(|c| !c.is_empty()).count(),
                    label: STRUCTURAL_ENVELOPE_LABEL,
                });
            }
            t -= 1;
            choice[t] += 1;
            if choice[t] <= unseen[t].len() {
                break;
            }
            choice[t] = 0;
        }
    }
}
