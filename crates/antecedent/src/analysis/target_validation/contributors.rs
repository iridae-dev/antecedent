//! Preserve the structural response contributors across refits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, ExecutionContext, ResponseValue, StudyResult, TargetRefitError,
    TemporalResponseMemberSelector, failed, unsupported,
};

/// Read the exact retained member of a whole-program result. Missing or repeated
/// identities are refused; a refit is never substituted with another completion.
pub(super) fn temporal_response_member_values(
    result: &StudyResult,
    selector: &TemporalResponseMemberSelector,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, CausalError> {
    let Some(posterior) = &result.temporal_posterior_witnesses else {
        return Err(CausalError::Unsupported {
            message: "temporal response has no retained completion witnesses",
        });
    };
    let mut visits = u64::try_from(posterior.members.len())
        .map_err(|_| CausalError::Resource { message: "temporal selector work overflow".into() })?;
    for member in &posterior.members {
        visits = visits
            .checked_add(u64::try_from(member.witnesses.entries.len()).map_err(|_| {
                CausalError::Resource { message: "temporal selector work overflow".into() }
            })?)
            .ok_or_else(|| CausalError::Resource {
                message: "temporal selector work overflow".into(),
            })?;
    }
    budget
        .charge_work(visits)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let mut matched = None;
    for member in &posterior.members {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal response selector" });
        }
        if member.outer_atom != selector.outer_atom {
            continue;
        }
        for witness in &member.witnesses.entries {
            if witness.horizon_index != selector.horizon_index
                || witness.horizon != selector.horizon
                || witness.completion_digest != selector.completion_digest
            {
                continue;
            }
            if matched.is_some() {
                return Err(CausalError::Unsupported {
                    message: "temporal response completion identity is ambiguous",
                });
            }
            let atom = member.structural.atoms.get(witness.atom_index).ok_or(
                CausalError::Unsupported {
                    message: "temporal response completion witness has no fitted atom",
                },
            )?;
            let values = match atom.value.as_ref() {
                Some(ResponseValue::Scalar(value)) => vec![*value],
                Some(ResponseValue::Surface { mean, .. }) if mean.len() == 1 => mean.to_vec(),
                Some(ResponseValue::Vector(values)) if values.len() == 1 => values.to_vec(),
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "temporal response completion has no point-level target",
                    });
                }
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(CausalError::Unsupported {
                    message: "temporal response completion target is not finite",
                });
            }
            matched = Some(values);
        }
    }
    matched.ok_or(CausalError::Unsupported {
        message: "temporal response refit omitted the original completion",
    })
}

// Static refits must keep every structural contributor, including nonidentified
// atoms: changing the conditioning mass would change the target being checked.
pub(super) fn check_static_response_contributors(
    before: &crate::result::StructuralResponseMixture,
    after: &crate::result::StructuralResponseMixture,
    source_bound_ordered: bool,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(), TargetRefitError> {
    let changed = || {
        unsupported(
            "refit changed structural contributors, mass, or coordinates",
            "refit the same identified structural contributors",
        )
    };
    if before.atoms.is_empty()
        || before.atoms.len() != after.atoms.len()
        || before.weight_basis != after.weight_basis
        || before.aggregation_policy != after.aggregation_policy
        || before.full_mass_scope != after.full_mass_scope
        || before.truncated_atoms != after.truncated_atoms
        || [
            before.identified_mass,
            before.unidentified_mass,
            before.unevaluable_mass,
            before.subsampled_out_mass,
        ]
        .iter()
        .zip([
            after.identified_mass,
            after.unidentified_mass,
            after.unevaluable_mass,
            after.subsampled_out_mass,
        ])
        .any(|(a, b)| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
    {
        return Err(changed());
    }
    let count = u64::try_from(before.atoms.len()).map_err(|_| changed())?;
    if source_bound_ordered {
        budget
            .charge_work(count)
            .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
        for (atom, other) in before.atoms.iter().zip(&after.atoms) {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled {
                    stage: "static response contributors",
                }));
            }
            if !atom.weight.is_finite()
                || atom.weight < 0.0
                || atom.graph_key != other.graph_key
                || atom.weight.to_bits() != other.weight.to_bits()
                || atom.status != other.status
                || !same_static_response_layout(atom, other)
            {
                return Err(changed());
            }
        }
        return Ok(());
    }
    let work = count.checked_mul(count).and_then(|n| n.checked_mul(2)).ok_or_else(changed)?;
    budget
        .charge_work(work)
        .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
    for atom in &before.atoms {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "static response contributors" }));
        }
        if !atom.weight.is_finite() || atom.weight < 0.0 {
            return Err(changed());
        }
        let mut original_count = 0usize;
        for other in &before.atoms {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled {
                    stage: "static response contributors",
                }));
            }
            if other.graph_key == atom.graph_key {
                original_count += 1;
            }
        }
        if original_count != 1 {
            return Err(changed());
        }
        let mut matched = None;
        for other in &after.atoms {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled {
                    stage: "static response contributors",
                }));
            }
            if other.graph_key == atom.graph_key && matched.replace(other).is_some() {
                return Err(changed());
            }
        }
        let other = matched.ok_or_else(changed)?;
        if atom.weight.to_bits() != other.weight.to_bits()
            || atom.status != other.status
            || !same_static_response_layout(atom, other)
        {
            return Err(changed());
        }
    }
    Ok(())
}

// Both executions must retain the exact immutable posterior mask source;
// opaque keys alone cannot prove the identity of wider adjacency masks.
pub(super) fn same_admg_posterior_source(
    posterior: &antecedent_discovery::GraphPosterior,
    original: Option<&crate::result::StaticPosteriorSourceWitness>,
    refitted: Option<&crate::result::StaticPosteriorSourceWitness>,
) -> bool {
    original
        .and_then(|w| w.source.upgrade())
        .zip(refitted.and_then(|w| w.source.upgrade()))
        .is_some_and(|(first, second)| {
            Arc::ptr_eq(&first, posterior.atoms()) && Arc::ptr_eq(&second, posterior.atoms())
        })
}

pub(super) fn exact_admg_contributors(
    posterior: &antecedent_discovery::GraphPosterior,
    before: &crate::result::StructuralResponseMixture,
    after: &crate::result::StructuralResponseMixture,
    source_matches: bool,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<bool, TargetRefitError> {
    if !matches!(
        posterior.atom_kind(),
        antecedent_discovery::GraphPosteriorAtomKind::Admg
            | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
    ) || (!source_matches && posterior.n_vars() > 8)
        || posterior.max_lag().is_some()
        || posterior.lag_masks().is_some()
        || posterior.lag_bidirected_masks().is_some()
        || posterior.mark_atoms().is_some()
        || posterior.atoms().len() != before.atoms.len()
        || posterior.atoms().len() != after.atoms.len()
        || posterior.atoms().len() != posterior.weights().len()
        || posterior.atoms().len() != posterior.graph_keys().len()
    {
        return Ok(false);
    }
    let count = u64::try_from(posterior.atoms().len()).map_err(|_| {
        failed(&CausalError::Resource { message: "ADMG contributor count overflow".into() })
    })?;
    budget
        .charge_work(count)
        .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
    for ((((mask, key), weight), original), refitted) in posterior
        .atoms()
        .iter()
        .zip(posterior.graph_keys().iter())
        .zip(posterior.weights().iter())
        .zip(before.atoms.iter())
        .zip(after.atoms.iter())
    {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "ADMG response contributors" }));
        }
        if (!source_matches && (mask.words().len() > 4 || mask.to_word() != Some(*key)))
            || original.graph_key != *key
            || refitted.graph_key != *key
            || !weight.is_finite()
            || original.weight.to_bits() != weight.to_bits()
            || refitted.weight.to_bits() != weight.to_bits()
            || original.status != refitted.status
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn same_static_class_witnesses(
    original: &StudyResult,
    refitted: &StudyResult,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<bool, TargetRefitError> {
    let (Some(a), Some(b)) =
        (original.static_class_witnesses.as_ref(), refitted.static_class_witnesses.as_ref())
    else {
        return Ok(false);
    };
    compare_static_class_witnesses(a, b, budget, ctx)
}

pub(super) fn compare_static_class_witnesses(
    a: &crate::result::StaticClassWitnessSet,
    b: &crate::result::StaticClassWitnessSet,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<bool, TargetRefitError> {
    let same_cache = a
        .identification_cache
        .upgrade()
        .zip(b.identification_cache.upgrade())
        .is_some_and(|(left, right)| Arc::ptr_eq(&left, &right));
    let same_atoms = a
        .source_atoms
        .upgrade()
        .zip(b.source_atoms.upgrade())
        .is_some_and(|(left, right)| Arc::ptr_eq(&left, &right));
    let same_marks = match (&a.source_marks, &b.source_marks) {
        (None, None) => true,
        (Some(left), Some(right)) => left
            .upgrade()
            .zip(right.upgrade())
            .is_some_and(|(left, right)| Arc::ptr_eq(&left, &right)),
        _ => false,
    };
    if !same_cache
        || !same_atoms
        || !same_marks
        || a.atom_kind != b.atom_kind
        || a.n_vars != b.n_vars
        || a.atoms.len() != b.atoms.len()
        || a.samples.len() != b.samples.len()
    {
        return Ok(false);
    }
    let sample_count = u64::try_from(a.samples.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "class sample count overflow".into() })
    })?;
    budget
        .charge_work(sample_count)
        .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
    for (left, right) in a.samples.iter().zip(&b.samples) {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "class response contributors" }));
        }
        if left != right {
            return Ok(false);
        }
    }
    let count = u64::try_from(a.atoms.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "class witness count overflow".into() })
    })?;
    budget
        .charge_work(count)
        .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
    for (left, right) in a.atoms.iter().zip(&b.atoms) {
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled { stage: "class response contributors" }));
        }
        if left.graph_key != right.graph_key
            || left.outer_weight != right.outer_weight
            || left.completions.entries.len() != right.completions.entries.len()
        {
            return Ok(false);
        }
        let work = u64::try_from(left.completions.entries.len()).map_err(|_| {
            failed(&CausalError::Resource { message: "class completion count overflow".into() })
        })?;
        budget
            .charge_work(work)
            .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
        for (a, b) in left.completions.entries.iter().zip(&right.completions.entries) {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled {
                    stage: "class response contributors",
                }));
            }
            if a != b {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn same_static_response_layout(
    a: &crate::result::StructuralResponseAtom,
    b: &crate::result::StructuralResponseAtom,
) -> bool {
    match (&a.response, &b.response) {
        (None, None) => {}
        (Some(x), Some(y))
            if x.estimand == y.estimand
                && x.support.status == y.support.status
                && x.support.point_status == y.support.point_status => {}
        _ => return false,
    }
    match (&a.value, &b.value) {
        (None, None) | (Some(ResponseValue::Scalar(_)), Some(ResponseValue::Scalar(_))) => true,
        (Some(ResponseValue::Vector(x)), Some(ResponseValue::Vector(y))) => x.len() == y.len(),
        (
            Some(ResponseValue::Surface { grid: x, dimension: dx, mean: mx }),
            Some(ResponseValue::Surface { grid: y, dimension: dy, mean: my }),
        ) => {
            dx == dy
                && mx.len() == my.len()
                && x.len() == y.len()
                && x.iter().zip(y.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (
            Some(ResponseValue::Jacobian { outcomes: ox, treatments: tx, values: x }),
            Some(ResponseValue::Jacobian { outcomes: oy, treatments: ty, values: y }),
        ) => ox == oy && tx == ty && x.len() == y.len(),
        _ => false,
    }
}

fn same_temporal_point_layout(
    a: &crate::result::StructuralResponseAtom,
    b: &crate::result::StructuralResponseAtom,
) -> bool {
    if a.status != b.status {
        return false;
    }
    match (&a.response, &b.response) {
        (Some(x), Some(y))
            if x.support.status == y.support.status
                && x.support.point_status == y.support.point_status => {}
        _ => return false,
    }
    match (&a.value, &b.value) {
        (Some(ResponseValue::Scalar(_)), Some(ResponseValue::Scalar(_))) => true,
        (Some(ResponseValue::Vector(x)), Some(ResponseValue::Vector(y))) => x.len() == y.len(),
        (
            Some(ResponseValue::Surface { grid: x, dimension: dx, mean: mx }),
            Some(ResponseValue::Surface { grid: y, dimension: dy, mean: my }),
        ) => {
            dx == dy
                && mx.len() == my.len()
                && x.len() == y.len()
                && x.iter().zip(y.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the target-specific branches preserve a single checked operation"
)]
pub(super) fn check_temporal_response_contributors(
    original: &StudyResult,
    refitted: &StudyResult,
    budget: &mut antecedent_core::TemporalBudget,
    include_zero: bool,
    ctx: &ExecutionContext,
) -> Result<(), TargetRefitError> {
    let eligible = |weight: f64| include_zero || weight > 0.0;
    let Some(before) = &original.temporal_posterior_witnesses else {
        return Ok(());
    };
    let expected = original.structural_response.as_ref().map_or(0, |mixture| {
        mixture.atoms.iter().filter(|atom| eligible(atom.weight) && atom.value.is_some()).count()
    });
    let before_count = before.members.iter().filter(|member| eligible(member.outer_weight)).count();
    if expected != before_count || expected == 0 {
        return Err(unsupported(
            "original result did not retain every contributing posterior member",
            "refit with complete point-time completion witnesses",
        ));
    }
    let after = refitted.temporal_posterior_witnesses.as_ref().ok_or_else(|| {
        unsupported(
            "refit omitted temporal response witnesses",
            "refit with the same completion coverage",
        )
    })?;
    if before_count != after.members.iter().filter(|member| eligible(member.outer_weight)).count() {
        return Err(unsupported(
            "refit changed the contributing posterior members",
            "refit with the same completion coverage",
        ));
    }
    let count = u64::try_from(before.members.len()).map_err(|_| {
        failed(&CausalError::Resource { message: "posterior witness count overflow".into() })
    })?;
    let work = count.checked_mul(count).and_then(|n| n.checked_mul(2)).ok_or_else(|| {
        failed(&CausalError::Resource { message: "posterior witness comparison overflow".into() })
    })?;
    budget
        .charge_work(work)
        .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
    for a in before.members.iter().filter(|member| eligible(member.outer_weight)) {
        let point_atoms = a
            .structural
            .atoms
            .iter()
            .filter(|atom| eligible(atom.weight) && atom.value.is_some())
            .count();
        let captured =
            a.witnesses.entries.iter().filter(|witness| eligible(witness.weight)).count();
        if captured == 0 || point_atoms != captured {
            return Err(unsupported(
                "original member did not retain every fitted completion",
                "refit with complete point-time completion witnesses",
            ));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(failed(&CausalError::Cancelled {
                stage: "temporal response contributors",
            }));
        }
        if before
            .members
            .iter()
            .filter(|member| eligible(member.outer_weight) && member.outer_atom == a.outer_atom)
            .count()
            != 1
        {
            return Err(unsupported(
                "ambiguous original posterior member",
                "refit with unique member identities",
            ));
        }
        let mut matches = after
            .members
            .iter()
            .filter(|member| eligible(member.outer_weight) && member.outer_atom == a.outer_atom);
        let b = matches.next().ok_or_else(|| {
            unsupported(
                "refit omitted a posterior member",
                "refit with the same completion coverage",
            )
        })?;
        if matches.next().is_some()
            || a.outer_weight.to_bits() != b.outer_weight.to_bits()
            || captured
                != b.witnesses.entries.iter().filter(|witness| eligible(witness.weight)).count()
        {
            return Err(unsupported(
                "refit changed a posterior member or its weight",
                "refit with unique member identities and weights",
            ));
        }
        let n = u64::try_from(a.witnesses.entries.len()).map_err(|_| {
            failed(&CausalError::Resource { message: "completion witness count overflow".into() })
        })?;
        budget
            .charge_work(n.checked_mul(n).and_then(|count| count.checked_mul(2)).ok_or_else(
                || {
                    failed(&CausalError::Resource {
                        message: "completion witness comparison overflow".into(),
                    })
                },
            )?)
            .map_err(|error| failed(&CausalError::Resource { message: error.to_string() }))?;
        for x in a.witnesses.entries.iter().filter(|witness| eligible(witness.weight)) {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled {
                    stage: "temporal response contributors",
                }));
            }
            let same = |y: &&crate::result::TemporalResponseWitness| {
                eligible(y.weight)
                    && y.horizon_index == x.horizon_index
                    && y.horizon == x.horizon
                    && y.completion_digest == x.completion_digest
            };
            if a.witnesses.entries.iter().filter(same).count() != 1 {
                return Err(unsupported(
                    "ambiguous original completion",
                    "refit with unique completion identities",
                ));
            }
            let mut matches = b.witnesses.entries.iter().filter(same);
            let y = matches.next().ok_or_else(|| {
                unsupported("refit omitted a completion", "refit with the same completion coverage")
            })?;
            if matches.next().is_some() || x.weight.to_bits() != y.weight.to_bits() {
                return Err(unsupported(
                    "refit changed a completion identity or weight",
                    "refit with the same member identities and weights",
                ));
            }
            let left = a.structural.atoms.get(x.atom_index).ok_or_else(|| {
                unsupported("original completion has no fitted atom", "refit the original program")
            })?;
            let right = b.structural.atoms.get(y.atom_index).ok_or_else(|| {
                unsupported("refit completion has no fitted atom", "refit the original program")
            })?;
            if !same_temporal_point_layout(left, right) {
                return Err(unsupported(
                    "refit changed completion coordinates, support, or identification",
                    "refit on the same supported coordinates",
                ));
            }
        }
    }
    if original.structural_response.as_ref().and_then(|v| v.aggregation_policy)
        != refitted.structural_response.as_ref().and_then(|v| v.aggregation_policy)
    {
        return Err(unsupported(
            "refit changed the structural aggregation policy",
            "refit under the same aggregation policy",
        ));
    }
    Ok(())
}
