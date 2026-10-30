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
    DETAIL_ENVELOPE_BUDGET, DETAIL_ENVELOPE_EMPTY, SampledTransportLimits, SampledWork,
    evaluate_detailed, refused,
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
    work: &mut SampledWork<'_>,
) -> Result<StructuralEnvelope, EstimationError> {
    let unseen: Vec<Vec<usize>> =
        tables.iter().map(|table| table.checked_unseen_cells(work)).collect::<Result<_, _>>()?;
    let mut members: u64 = 1;
    let mut tables_with_unseen_cells = 0;
    for cells in &unseen {
        work.charge(0)?;
        tables_with_unseen_cells += usize::from(!cells.is_empty());
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
    let mut flavors = Vec::with_capacity(tables.len());
    let mut choice = Vec::with_capacity(tables.len());
    while choice.len() < tables.len() {
        work.charge(0)?;
        let end = choice.len().saturating_add(512).min(tables.len());
        choice.resize(end, 0usize);
        flavors.resize(end, LawFlavor::Model);
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut defined = 0u64;
    let mut undefined = 0u64;
    loop {
        let mut member_work = SampledWork { remaining: limits.evaluation.operations, ctx };
        let mut trial = Vec::with_capacity(tables.len());
        for t in 0..tables.len() {
            member_work.charge(probabilities[t].len().saturating_mul(2))?;
            let mut p = Vec::with_capacity(probabilities[t].len());
            for chunk in probabilities[t].chunks(512) {
                member_work.charge(0)?;
                p.extend_from_slice(chunk);
            }
            if choice[t] > 0 {
                let cell = unseen[t][choice[t] - 1];
                for chunk in p.chunks_mut(512) {
                    member_work.charge(0)?;
                    for q in chunk {
                        *q *= 1.0 - unseen_mass;
                    }
                }
                p[cell] += unseen_mass;
            }
            trial.push(p);
        }
        match evaluate_detailed(derivation, tables, &trial, &flavors, limits, ctx, &mut member_work)
            .map(|(point, _)| point.probability)
        {
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
                    tables_with_unseen_cells,
                    label: STRUCTURAL_ENVELOPE_LABEL,
                });
            }
            member_work.charge(0)?;
            t -= 1;
            choice[t] += 1;
            if choice[t] <= unseen[t].len() {
                break;
            }
            choice[t] = 0;
        }
    }
}
